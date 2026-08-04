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
/// <see cref="TokenCorpus"/>, build a <see cref="GptModel"/> - fresh, from
/// CLI-supplied architecture flags, or loaded from an existing checkpoint
/// via TASK-040's <c>--resume-from-checkpoint</c> - train it, and
/// checkpoint the result.
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
            stdout.WriteLine("  [--scratch-dir <dir>] [--optimised | --gpu [--gpu-allow-cpu-fallback] [--gpu-resident-weights]]");
            stdout.WriteLine("  [--resume-from-checkpoint <path>]");
            stdout.WriteLine();
            stdout.WriteLine("Trains a fresh GptModel from scratch on the given corpus, using a tokeniser");
            stdout.WriteLine("vocabulary already trained via the Tokeniser CLI, and saves the result as a");
            stdout.WriteLine("checkpoint. Directories are expanded to their *.txt files.");
            stdout.WriteLine();
            stdout.WriteLine("--resume-from-checkpoint <path> (TASK-040) loads an existing checkpoint and");
            stdout.WriteLine("continues training it on the given corpus, instead of building a fresh model -");
            stdout.WriteLine("the way to train on more data than fits in one run, in bounded-RAM chunks, with");
            stdout.WriteLine("the model file compounding over successive runs (see README.md stage 12).");
            stdout.WriteLine("The model's architecture (--embedding-dim/--layers/--heads/--context-length)");
            stdout.WriteLine("comes from the checkpoint itself and cannot be overridden - specifying any of");
            stdout.WriteLine("those flags together with --resume-from-checkpoint is an error, not a silent");
            stdout.WriteLine("reshape. The optimizer's per-parameter moment estimates (AdamW's first/second");
            stdout.WriteLine("moment) are NOT part of the checkpoint format and restart from zero on every");
            stdout.WriteLine("resume - a deliberate, documented trade-off (see README.md stage 12), not an");
            stdout.WriteLine("oversight.");
            stdout.WriteLine();
            stdout.WriteLine("--gpu (TASK-033) selects the ILGPU-backed matmul path and requires a genuine");
            stdout.WriteLine("CUDA/OpenCL accelerator by default - it refuses with a clear error rather than");
            stdout.WriteLine("silently training on ILGPU's CPU accelerator. Pass --gpu-allow-cpu-fallback to");
            stdout.WriteLine("accept that CPU accelerator anyway (useful for exercising the mechanism on a");
            stdout.WriteLine("machine without a working GPU driver). --gpu and --optimised select different");
            stdout.WriteLine("backends and cannot be combined.");
            stdout.WriteLine();
            stdout.WriteLine("--gpu-resident-weights (TASK-036, requires --gpu) keeps every model parameter");
            stdout.WriteLine("device-resident for the whole run, so a forward pass's matmuls stop re-uploading");
            stdout.WriteLine("them every step - backward and the optimizer's update run their own");
            stdout.WriteLine("device-resident kernels too (TASK-037/038/039), but measured slower than the");
            stdout.WriteLine("non-resident path at this project's toy model sizes, not faster - off by");
            stdout.WriteLine("default because of this. See README.md stage 11.");
            return 1;
        }

        string vocabPath = args[0];
        string checkpointPath = args[1];

        int embeddingDim = 128, layers = 4, heads = 4, contextLength = 64, steps = 2000, batchSize = 8;
        float learningRate = 3e-4f, weightDecay = 0.01f;
        string scratchDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "pretrain-scratch");
        bool optimised = false;
        bool gpu = false;
        bool gpuAllowCpuFallback = false;
        bool gpuResidentWeights = false;
        string? resumeFromCheckpointPath = null;
        bool embeddingDimSpecified = false, layersSpecified = false, headsSpecified = false, contextLengthSpecified = false;

        var positionalArgs = new List<string>();
        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--embedding-dim" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedEmbeddingDim):
                    embeddingDim = parsedEmbeddingDim;
                    embeddingDimSpecified = true;
                    i++;
                    break;
                case "--layers" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedLayers):
                    layers = parsedLayers;
                    layersSpecified = true;
                    i++;
                    break;
                case "--heads" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedHeads):
                    heads = parsedHeads;
                    headsSpecified = true;
                    i++;
                    break;
                case "--context-length" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedContextLength):
                    contextLength = parsedContextLength;
                    contextLengthSpecified = true;
                    i++;
                    break;
                case "--resume-from-checkpoint" when i + 1 < args.Length:
                    resumeFromCheckpointPath = args[i + 1];
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
                case "--gpu":
                    gpu = true;
                    break;
                case "--gpu-allow-cpu-fallback":
                    gpuAllowCpuFallback = true;
                    break;
                case "--gpu-resident-weights":
                    gpuResidentWeights = true;
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

        if (optimised && gpu)
        {
            stderr.WriteLine("Specify either --optimised or --gpu, not both - they select different Tensor backends.");
            return 1;
        }

        if (gpuAllowCpuFallback && !gpu)
        {
            stderr.WriteLine("--gpu-allow-cpu-fallback only makes sense together with --gpu.");
            return 1;
        }

        if (gpuResidentWeights && !gpu)
        {
            stderr.WriteLine("--gpu-resident-weights only makes sense together with --gpu.");
            return 1;
        }

        if (resumeFromCheckpointPath is not null && (embeddingDimSpecified || layersSpecified || headsSpecified || contextLengthSpecified))
        {
            stderr.WriteLine("--embedding-dim/--layers/--heads/--context-length cannot be combined with " +
                "--resume-from-checkpoint - the model's architecture comes from the checkpoint itself.");
            return 1;
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
        else if (gpu)
        {
            // Preflight, before any training work starts: refuses with a
            // clear error here (TASK-031's GpuContext.GetAccelerator) rather
            // than silently running on ILGPU's CPU accelerator while
            // claiming to demonstrate GPU execution, unless the user
            // explicitly said that's fine via --gpu-allow-cpu-fallback.
            try
            {
                GpuContext.GetAccelerator(gpuAllowCpuFallback);
            }
            catch (InvalidOperationException ex)
            {
                stderr.WriteLine(ex.Message);
                return 1;
            }
            Tensor.Tensor.Backend = TensorBackend.Gpu;
        }

        GptModel model;
        if (resumeFromCheckpointPath is not null)
        {
            // TASK-040: continue an existing checkpoint rather than
            // starting fresh - the architecture (including context
            // length/MaxSequenceLength) comes entirely from the
            // checkpoint, validated above to not be overridden. AdamW's
            // moment estimates are NOT part of the checkpoint format and
            // restart from zero here - a deliberate, documented trade-off
            // (see README.md stage 12), not an oversight.
            try
            {
                model = ModelCheckpoint.Load(resumeFromCheckpointPath);
            }
            catch (Exception ex) when (ex is IOException or InvalidOperationException)
            {
                stderr.WriteLine($"Failed to load checkpoint to resume: {ex.Message}");
                return 1;
            }
            if (model.VocabSize != tokeniser.VocabSize)
            {
                stderr.WriteLine($"Checkpoint's vocab size ({model.VocabSize}) doesn't match the loaded tokeniser's ({tokeniser.VocabSize}) - they must come from the same tokeniser.");
                return 1;
            }
            contextLength = model.MaxSequenceLength;
        }
        else
        {
            model = new GptModel(
                vocabSize: tokeniser.VocabSize,
                embeddingDim: embeddingDim,
                numLayers: layers,
                numHeads: heads,
                maxSequenceLength: contextLength);
        }

        stdout.WriteLine($"Bulk-encoding {corpusFiles.Count} corpus file(s)...");
        using var encoded = tokeniser.EncodeBulk(corpusFiles, scratchDirectory);
        using var corpus = new TokenCorpus(encoded, scratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength);

        stdout.WriteLine($"Training a {model.NumLayers}-layer, {model.EmbeddingDim}-dim GptModel for {steps} steps " +
            $"(batch size {batchSize}, context length {contextLength}, {corpus.Length} tokens)...");

        if (gpuResidentWeights)
        {
            // TASK-036: move every parameter's storage onto the GPU once,
            // in place (Tensor.MoveToGpuInPlace) - a forward pass's matmuls
            // then use each weight's existing device view directly
            // (TASK-035) instead of re-uploading it every step. Backward
            // and the optimizer's update below run their own
            // device-resident kernels too (TASK-037/038/039), but measured
            // slower overall, not faster - see this flag's own --help text
            // and README.md stage 11.
            foreach (var parameter in model.Parameters())
            {
                parameter.Value.MoveToGpuInPlace();
            }
        }

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
