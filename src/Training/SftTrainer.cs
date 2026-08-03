using Model;
using Tensor;
using TensorValue = Tensor.Tensor;

namespace Training;

/// <summary>
/// Fine-tunes a <see cref="GptModel"/> on (instruction, response) pairs
/// (TASK-016), with the loss restricted to response tokens only (see
/// <see cref="CrossEntropyLoss.ComputeMasked"/>). A sibling to
/// <see cref="Trainer"/> (TASK-012) rather than a generalisation of it:
/// the two operate over genuinely different data shapes (a continuous
/// token stream sliced into sliding windows there, standalone
/// instruction/response sequences here) and different losses, so sharing
/// an abstraction would cost more in indirection than it would save in
/// duplication - both are thin wrappers around the same underlying
/// forward/backward/optimizer-step machinery regardless.
///
/// Callers are expected to construct <paramref name="model"/> via
/// <see cref="ModelCheckpoint.Load"/> (fine-tuning starts from a
/// pretrained model, not a random init) and, once done, save the result
/// via <see cref="ModelCheckpoint.Save"/> to a *different* path than the
/// base checkpoint - this class never touches checkpointing itself, the
/// same way <see cref="Trainer"/> doesn't. Conventionally uses a smaller
/// learning rate than pretraining (a tenth or less is a common starting
/// point) - configure that on the <see cref="IOptimizer"/> passed in, not
/// here.
/// </summary>
public sealed class SftTrainer
{
    private readonly GptModel _model;
    private readonly IReadOnlyList<SftTokenizedExample> _examples;
    private readonly IOptimizer _optimizer;

    public SftTrainer(GptModel model, IReadOnlyList<SftTokenizedExample> examples, IOptimizer optimizer)
    {
        if (examples.Count == 0)
        {
            throw new ArgumentException("SFT dataset must contain at least one example.", nameof(examples));
        }

        _model = model;
        _examples = examples;
        _optimizer = optimizer;
    }

    /// <summary>
    /// Runs one gradient-accumulated step over <paramref name="batchSize"/>
    /// examples starting at <paramref name="startIndex"/> (wrapping around
    /// the dataset if it runs past the end) - mirrors <see cref="Trainer.Step"/>'s
    /// batch-average gradient scaling (each example's loss is scaled by
    /// 1/batchSize before its Backward(), so the accumulated gradient ends
    /// up averaged rather than summed). Returns the batch's average masked
    /// loss.
    /// </summary>
    public float Step(int batchSize, int startIndex)
    {
        var indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            indices[i] = (startIndex + i) % _examples.Count;
        }
        return StepOn(indices);
    }

    /// <summary>
    /// Runs one gradient-accumulated step over exactly the examples at
    /// <paramref name="exampleIndices"/> (in the given order, no
    /// wraparound) - the shared core both <see cref="Step"/> and
    /// <see cref="RunEpochs"/> (TASK-030) build on, since an epoch's final
    /// batch is often smaller than a full batch size and averaging by the
    /// *actual* count in the batch (not a fixed denominator) is what makes
    /// that batch's gradient scale correctly.
    /// </summary>
    private float StepOn(IReadOnlyList<int> exampleIndices)
    {
        _optimizer.ZeroGrad();

        float totalLoss = 0f;
        var inverseBatchSize = new Variable(TensorValue.FromValues([1f / exampleIndices.Count], [1]));

        foreach (int index in exampleIndices)
        {
            var example = _examples[index];
            var logits = _model.Forward(example.InputIds);
            var loss = CrossEntropyLoss.ComputeMasked(logits, example.TargetIds, example.ResponseMask);
            totalLoss += loss.Value.ToArray()[0];

            var scaledLoss = loss.Multiply(inverseBatchSize);
            scaledLoss.Backward();
        }

        _optimizer.Step();
        return totalLoss / exampleIndices.Count;
    }

    /// <summary>
    /// Runs <paramref name="steps"/> training steps, advancing sequentially
    /// through the dataset (wrapping around) rather than resampling
    /// randomly each step - deterministic and simple, appropriate for a
    /// typically small, curated SFT dataset (unlike TASK-010's large
    /// continuous pretraining corpus, where <see cref="BatchSampler"/>
    /// samples random windows because there are far too many to cycle
    /// through in order). <paramref name="onStep"/>, if given, is invoked
    /// after each step with (stepIndex, averageLoss).
    /// </summary>
    public void Run(int steps, int batchSize, Action<int, float>? onStep = null)
    {
        int index = 0;
        for (int step = 0; step < steps; step++)
        {
            float loss = Step(batchSize, index);
            index = (index + batchSize) % _examples.Count;
            onStep?.Invoke(step, loss);
        }
    }

    /// <summary>
    /// Runs <paramref name="epochs"/> full, independently-shuffled passes
    /// over the dataset (TASK-030) instead of a fixed step count the
    /// caller has to size against dataset length themselves - each epoch
    /// is <c>ceil(datasetSize / batchSize)</c> steps, so the total amount
    /// of training scales automatically with how much data there is,
    /// rather than requiring <see cref="Run"/>'s caller to already know
    /// "enough steps" for their dataset. Shuffling (a fresh
    /// <see cref="Random"/>-driven permutation per epoch, Fisher-Yates) is
    /// genuinely new here - <see cref="Run"/>'s sequential wraparound order
    /// is deterministic and fine for a tiny, curated demo dataset, but a
    /// real dataset of hundreds or thousands of examples benefits from not
    /// seeing the same fixed example order every epoch. The final batch of
    /// an epoch is often smaller than a full batch (when dataset size
    /// isn't a multiple of batchSize) - <see cref="StepOn"/> averages by
    /// that batch's actual size, not a fixed denominator, so its gradient
    /// scale is still correct. <paramref name="onStep"/>, if given, is
    /// invoked after each step with (epochIndex, globalStepIndex,
    /// averageLoss).
    /// </summary>
    public void RunEpochs(int epochs, int batchSize, Random random, Action<int, int, float>? onStep = null)
    {
        int globalStep = 0;
        var order = new int[_examples.Count];
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(order, random);

            for (int start = 0; start < order.Length; start += batchSize)
            {
                int count = Math.Min(batchSize, order.Length - start);
                var batchIndices = new ArraySegment<int>(order, start, count);
                float loss = StepOn(batchIndices);
                onStep?.Invoke(epoch, globalStep, loss);
                globalStep++;
            }
        }
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }
}
