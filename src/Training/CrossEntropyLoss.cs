using Tensor;
using TensorValue = Tensor.Tensor;

namespace Training;

/// <summary>
/// Next-token-prediction loss: the average negative log-probability the
/// model assigned to the actual next token at each position. Minimising
/// this is exactly "make the model assign higher probability to what
/// really came next" - the training objective for a decoder-only language
/// model.
/// </summary>
public static class CrossEntropyLoss
{
    /// <summary>
    /// <paramref name="logits"/> is [sequenceLength, vocabSize] (a
    /// <c>GptModel.Forward</c> output); <paramref name="targetTokenIds"/>
    /// has one entry per position (a <see cref="TrainingExample.Target"/>).
    /// Returns a [1] scalar loss.
    /// </summary>
    public static Variable Compute(Variable logits, int[] targetTokenIds)
    {
        int axis = logits.Value.Shape.Length - 1;

        // log_softmax(x) = x - logsumexp(x), computed this way (rather than
        // Softmax().Log()) so the normalising constant is worked out in
        // log-space directly instead of round-tripping through a softmax
        // value that could already have underflowed to 0. logsumexp(x) is
        // computed via the standard shifted form - max(x) + log(sum(exp(x -
        // max(x)))) - so every Exp argument stays <= 0 (TASK-017's "safe
        // softmax" trick), avoiding overflow for large-magnitude logits.
        // maxValue is wrapped as a constant (no parent op), which
        // deliberately stops gradient flowing through the max itself -
        // standard practice, and correct here since d(logsumexp)/dx_i =
        // softmax(x)_i regardless of the shift.
        var maxValue = new Variable(logits.Value.Max(axis, keepDims: true));
        var shifted = logits.Subtract(maxValue);
        var logSumExpShifted = shifted.Exp().Sum(axis, keepDims: true).Log();
        var logSoftmax = shifted.Subtract(logSumExpShifted);

        var targetLogProbs = logSoftmax.GatherColumns(targetTokenIds);
        return targetLogProbs.Negate().Mean(axis: 0, keepDims: true);
    }
}
