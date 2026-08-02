using Model;
using Tensor;
using TensorValue = Tensor.Tensor;

namespace Training;

/// <summary>
/// Wires TASK-009 (GptModel), TASK-010 (BatchSampler), and TASK-011
/// (CrossEntropyLoss + IOptimizer) into an actual training loop:
/// forward pass, loss, backward pass, optimizer step, repeated.
///
/// GptModel has no batch dimension of its own (see BatchSampler's doc
/// comment), so a "batch" here means accumulating gradients over several
/// individual forward/backward passes before a single optimizer step -
/// each example's loss is scaled by 1/batchSize before its Backward()
/// call, so the accumulated gradient ends up averaged over the batch
/// rather than summed (keeping the effective step size independent of
/// batch size).
/// </summary>
public sealed class Trainer
{
    private readonly GptModel _model;
    private readonly BatchSampler _sampler;
    private readonly IOptimizer _optimizer;

    public Trainer(GptModel model, BatchSampler sampler, IOptimizer optimizer)
    {
        _model = model;
        _sampler = sampler;
        _optimizer = optimizer;
    }

    /// <summary>Runs one training step (a full batch) and returns the batch's average loss.</summary>
    public float Step(int batchSize)
    {
        _optimizer.ZeroGrad();

        float totalLoss = 0f;
        var inverseBatchSize = TensorValue.FromValues([1f / batchSize], [1]);

        foreach (var example in _sampler.SampleBatch(batchSize))
        {
            var logits = _model.Forward(example.Input);
            var loss = CrossEntropyLoss.Compute(logits, example.Target);
            totalLoss += loss.Value.ToArray()[0];

            var scaledLoss = loss.Multiply(new Variable(inverseBatchSize));
            scaledLoss.Backward();
        }

        _optimizer.Step();
        return totalLoss / batchSize;
    }

    /// <summary>
    /// Runs <paramref name="steps"/> training steps. <paramref name="onStep"/>,
    /// if given, is invoked after each step with (stepIndex, averageLoss) -
    /// logging/checkpointing are the caller's concern, not this class's
    /// (mirrors BpeTokeniser.Train's onMerge callback).
    /// </summary>
    public void Run(int steps, int batchSize, Action<int, float>? onStep = null)
    {
        for (int step = 0; step < steps; step++)
        {
            float loss = Step(batchSize);
            onStep?.Invoke(step, loss);
        }
    }
}
