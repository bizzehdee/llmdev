using Model;
using Tensor;
using Tokeniser;
using Training;

namespace Sft;

/// <summary>
/// The instruction-tuning (SFT) CLI's actual logic, factored out of
/// <c>Program.cs</c> the same way <c>Pretrain.PretrainCli</c>/
/// <c>Chat.ChatCli</c> are - top-level statements can't be invoked from a
/// test project.
///
/// TASK-026: the same CLI gap TASK-025 filled for pretraining, but for
/// TASK-016's instruction tuning - fine-tuning a pretrained checkpoint on
/// an (instruction, response) dataset was previously library-only. Wires
/// TASK-016's <see cref="SftTrainer"/> up end to end: load a *pretrained*
/// checkpoint and its tokeniser vocab, load a JSON Lines instruction/
/// response dataset (<see cref="SftDataset.Load"/>), fine-tune, and save
/// the result to a *different* checkpoint path - the base pretrained
/// checkpoint is never overwritten, and this CLI refuses to try.
/// </summary>
public static class SftCli
{
    // TASK-030: tuned for a real few-hundred-to-few-thousand-example
    // dataset, not the README's 6-example demo (which needs a far larger
    // --epochs, since one shuffled pass over 6 examples is barely any
    // training at all).
    private const int DefaultEpochs = 3;

    public static int Run(string[] args, TextWriter stdout, TextWriter stderr)
    {
        if (args.Length < 4)
        {
            stdout.WriteLine("Usage: Sft <base-checkpoint-path> <vocab-path> <dataset-path> <output-checkpoint-path>");
            stdout.WriteLine("  [--epochs <n> | --steps <n>] [--batch-size <n>] [--learning-rate <f>] [--weight-decay <f>] [--optimised]");
            stdout.WriteLine();
            stdout.WriteLine("Fine-tunes a pretrained GptModel checkpoint on an (instruction, response) JSON Lines");
            stdout.WriteLine("dataset (one {\"instruction\": \"...\", \"response\": \"...\"} object per line), with the");
            stdout.WriteLine("loss restricted to response tokens only, and saves the result to a separate checkpoint -");
            stdout.WriteLine("the base checkpoint is never overwritten.");
            stdout.WriteLine();
            stdout.WriteLine("--epochs (default 3) runs that many full, shuffled passes over the dataset - the number");
            stdout.WriteLine("of steps scales with dataset size automatically. --steps is a lower-level escape hatch:");
            stdout.WriteLine("a fixed step count, sequential (unshuffled) example order, no notion of an epoch. Only");
            stdout.WriteLine("one of the two may be given.");
            return 1;
        }

        string baseCheckpointPath = args[0];
        string vocabPath = args[1];
        string datasetPath = args[2];
        string outputCheckpointPath = args[3];

        // A tenth of Pretrain's 3e-4 default, per TASK-016's own guidance
        // that SFT conventionally uses a smaller learning rate than
        // pretraining.
        int? steps = null;
        int? epochs = null;
        int batchSize = 8;
        float learningRate = 3e-5f, weightDecay = 0.01f;
        bool optimised = false;

        for (int i = 4; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--steps" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedSteps):
                    steps = parsedSteps;
                    i++;
                    break;
                case "--epochs" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsedEpochs):
                    epochs = parsedEpochs;
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
                case "--optimised":
                    optimised = true;
                    break;
                default:
                    stderr.WriteLine($"Unrecognised or malformed option: {args[i]}");
                    return 1;
            }
        }

        if (steps.HasValue && epochs.HasValue)
        {
            stderr.WriteLine("Specify either --steps or --epochs, not both.");
            return 1;
        }

        if (string.Equals(Path.GetFullPath(outputCheckpointPath), Path.GetFullPath(baseCheckpointPath), StringComparison.Ordinal))
        {
            stderr.WriteLine("Refusing to fine-tune: <output-checkpoint-path> must differ from <base-checkpoint-path> -");
            stderr.WriteLine("the base pretrained checkpoint must never be overwritten.");
            return 1;
        }

        GptModel model;
        BpeTokeniser tokeniser;
        try
        {
            model = ModelCheckpoint.Load(baseCheckpointPath);
            tokeniser = BpeTokeniser.Load(vocabPath);
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException)
        {
            stderr.WriteLine($"Failed to load base checkpoint or vocabulary: {ex.Message}");
            return 1;
        }

        IReadOnlyList<SftTokenizedExample> examples;
        try
        {
            examples = SftDataset.Load(datasetPath, tokeniser);
        }
        catch (Exception ex) when (ex is IOException or InvalidOperationException or System.Text.Json.JsonException)
        {
            stderr.WriteLine($"Failed to load SFT dataset: {ex.Message}");
            return 1;
        }

        if (examples.Count == 0)
        {
            stderr.WriteLine("The SFT dataset contains no examples.");
            return 1;
        }

        if (optimised)
        {
            Tensor.Tensor.Backend = TensorBackend.Optimised;
        }

        var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: learningRate, weightDecay: weightDecay);
        var trainer = new SftTrainer(model, examples, optimizer);

        if (steps.HasValue)
        {
            stdout.WriteLine($"Fine-tuning on {examples.Count} example(s) for {steps} steps (batch size {batchSize})...");

            trainer.Run(steps.Value, batchSize, onStep: (step, loss) =>
            {
                if (step % 100 == 0 || step == steps.Value - 1)
                {
                    stdout.WriteLine($"step {step}: loss {loss:F4}");
                }
            });
        }
        else
        {
            int epochCount = epochs ?? DefaultEpochs;
            int stepsPerEpoch = (examples.Count + batchSize - 1) / batchSize;
            int totalSteps = stepsPerEpoch * epochCount;
            stdout.WriteLine($"Fine-tuning on {examples.Count} example(s) for {epochCount} epoch(s) " +
                $"({stepsPerEpoch} step(s)/epoch, {totalSteps} step(s) total, batch size {batchSize})...");

            trainer.RunEpochs(epochCount, batchSize, new Random(), onStep: (epoch, globalStep, loss) =>
            {
                if (globalStep % 100 == 0 || globalStep == totalSteps - 1)
                {
                    stdout.WriteLine($"step {globalStep}: loss {loss:F4}");
                }
            });
        }

        ModelCheckpoint.Save(model, outputCheckpointPath);
        stdout.WriteLine($"Saved fine-tuned checkpoint to {outputCheckpointPath}.");

        return 0;
    }
}
