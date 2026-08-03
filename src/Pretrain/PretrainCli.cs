using Model;
using Tensor;
using Tokeniser;
using Training;

namespace Pretrain;

/// <summary>
/// The pretraining CLI's actual logic, factored out of <c>Program.cs</c>
/// the same way <c>Tokeniser.TokeniserCli</c>/<c>Chat.ChatCli</c> are - top-level
/// statements can't be invoked from a test project.
///
/// TASK-025: every other stage that produces or consumes a model artifact
/// has a CLI (tokeniser training, chat) - pretraining a <see cref="GptModel"/>
/// from a corpus was previously library-only, runnable only by pasting the
/// README's C# snippet. This wires TASK-012's <see cref="Trainer"/> up end
/// to end: load a trained tokeniser vocab, bulk-encode the corpus
/// (TASK-018's <see cref="BpeTokeniser.EncodeBulk"/> - the large-corpus
/// path, not <see cref="BpeTokeniser.Encode"/>) into a
/// <see cref="TokenCorpus"/>, build a fresh <see cref="GptModel"/> from
/// CLI-supplied architecture flags, train it, and checkpoint the result.
/// </summary>
public static class PretrainCli
{
    public static int Run(string[] args, TextWriter stdout, TextWriter stderr)
    {
        if (args.Length < 3)
        {
            stdout.WriteLine("Usage: Pretrain <vocab-path> <output-checkpoint-path> <corpus-file-or-directory> [file-or-directory ...]");
            stdout.WriteLine("  [--embedding-dim <n>] [--layers <n>] [--heads <n>] [--context-length <n>]");
            stdout.WriteLine("  [--steps <n>] [--batch-size <n>] [--learning-rate <f>] [--weight-decay <f>]");
            stdout.WriteLine("  [--scratch-dir <dir>] [--optimised]");
            stdout.WriteLine();
            stdout.WriteLine("Trains a fresh GptModel from scratch on the given corpus, using a tokeniser");
            stdout.WriteLine("vocabulary already trained via the Tokeniser CLI, and saves the result as a");
            stdout.WriteLine("checkpoint. Directories are expanded to their *.txt files.");
            return 1;
        }

        string vocabPath = args[0];
        string checkpointPath = args[1];

        int embeddingDim = 128, layers = 4, heads = 4, contextLength = 64, steps = 2000, batchSize = 8;
        float learningRate = 3e-4f, weightDecay = 0.01f;
        string scratchDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "pretrain-scratch");
        bool optimised = false;

        var positionalArgs = new List<string>();
        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--embedding-dim" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedEmbeddingDim):
                    embeddingDim = parsedEmbeddingDim;
                    i++;
                    break;
                case "--layers" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedLayers):
                    layers = parsedLayers;
                    i++;
                    break;
                case "--heads" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedHeads):
                    heads = parsedHeads;
                    i++;
                    break;
                case "--context-length" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedContextLength):
                    contextLength = parsedContextLength;
                    i++;
                    break;
                case "--steps" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedSteps):
                    steps = parsedSteps;
                    i++;
                    break;
                case "--batch-size" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedBatchSize):
                    batchSize = parsedBatchSize;
                    i++;
                    break;
                case "--learning-rate" when i + 1 < args.Length && float.TryParse(args[i + 1], out float parsedLearningRate):
                    learningRate = parsedLearningRate;
                    i++;
                    break;
                case "--weight-decay" when i + 1 < args.Length && float.TryParse(args[i + 1], out float parsedWeightDecay):
                    weightDecay = parsedWeightDecay;
                    i++;
                    break;
                case "--scratch-dir" when i + 1 < args.Length:
                    scratchDirectory = args[i + 1];
                    i++;
                    break;
                case "--optimised":
                    optimised = true;
                    break;
                default:
                    if (args[i].StartsWith("--", StringComparison.Ordinal))
                    {
                        stderr.WriteLine($"Unrecognised or malformed option: {args[i]}");
                        return 1;
                    }
                    positionalArgs.Add(args[i]);
                    break;
            }
        }

        if (positionalArgs.Count == 0)
        {
            stderr.WriteLine("Provide at least one corpus file or directory.");
            return 1;
        }

        var corpusFiles = new List<string>();
        foreach (var input in positionalArgs)
        {
            if (Directory.Exists(input))
            {
                corpusFiles.AddRange(Directory.GetFiles(input, "*.txt"));
            }
            else if (File.Exists(input))
            {
                corpusFiles.Add(input);
            }
            else
            {
                stderr.WriteLine($"File or directory not found: {input}");
                return 1;
            }
        }

        if (corpusFiles.Count == 0)
        {
            stderr.WriteLine("No .txt files found in the given corpus input(s).");
            return 1;
        }

        BpeTokeniser tokeniser;
        try
        {
            tokeniser = BpeTokeniser.Load(vocabPath);
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException)
        {
            stderr.WriteLine($"Failed to load tokeniser vocabulary: {ex.Message}");
            return 1;
        }

        Directory.CreateDirectory(scratchDirectory);
        if (optimised)
        {
            Tensor.Tensor.Backend = TensorBackend.Optimised;
        }

        stdout.WriteLine($"Bulk-encoding {corpusFiles.Count} corpus file(s)...");
        using var encoded = tokeniser.EncodeBulk(corpusFiles, scratchDirectory);
        using var corpus = new TokenCorpus(encoded, scratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength);

        stdout.WriteLine($"Training a {layers}-layer, {embeddingDim}-dim GptModel for {steps} steps " +
            $"(batch size {batchSize}, context length {contextLength}, {corpus.Length} tokens)...");

        var model = new GptModel(
            vocabSize: tokeniser.VocabSize,
            embeddingDim: embeddingDim,
            numLayers: layers,
            numHeads: heads,
            maxSequenceLength: contextLength);

        var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: learningRate, weightDecay: weightDecay);
        var trainer = new Trainer(model, sampler, optimizer);

        trainer.Run(steps, batchSize, onStep: (step, loss) =>
        {
            if (step % 100 == 0 || step == steps - 1)
            {
                stdout.WriteLine($"step {step}: loss {loss:F4}");
            }
        });

        ModelCheckpoint.Save(model, checkpointPath);
        stdout.WriteLine($"Saved checkpoint to {checkpointPath}.");

        return 0;
    }
}
