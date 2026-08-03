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
        _optimizer.ZeroGrad();

        float totalLoss = 0f;
        var inverseBatchSize = new Variable(TensorValue.FromValues([1f / batchSize], [1]));

        for (int i = 0; i < batchSize; i++)
        {
            var example = _examples[(startIndex + i) % _examples.Count];
            var logits = _model.Forward(example.InputIds);
            var loss = CrossEntropyLoss.ComputeMasked(logits, example.TargetIds, example.ResponseMask);
            totalLoss += loss.Value.ToArray()[0];

            var scaledLoss = loss.Multiply(inverseBatchSize);
            scaledLoss.Backward();
        }

        _optimizer.Step();
        return totalLoss / batchSize;
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
}
