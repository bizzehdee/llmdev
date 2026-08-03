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
        var targetLogProbs = TargetLogProbs(logits, targetTokenIds);
        return targetLogProbs.Negate().Mean(axis: 0, keepDims: true);
    }

    /// <summary>
    /// TASK-016's instruction-tuning loss: same as <see cref="Compute"/>,
    /// but the mean is taken only over positions where
    /// <paramref name="responseMask"/> is true - a prompt/instruction
    /// token being predicted shouldn't count towards the loss, only a
    /// response token being predicted should (see
    /// <see cref="SftDataset.Tokenize"/> for how the mask lines up with
    /// <paramref name="targetTokenIds"/>). Plain <see cref="Compute"/>
    /// dividing by every position would dilute the loss whenever a prompt
    /// is long relative to its response.
    /// </summary>
    public static Variable ComputeMasked(Variable logits, int[] targetTokenIds, bool[] responseMask)
    {
        int maskedCount = responseMask.Count(m => m);
        if (maskedCount == 0)
        {
            throw new InvalidOperationException("responseMask must mark at least one position - nothing to compute a loss over.");
        }

        var targetLogProbs = TargetLogProbs(logits, targetTokenIds);

        var maskValues = responseMask.Select(m => m ? 1f : 0f).ToArray();
        var mask = new Variable(TensorValue.FromValues(maskValues, [responseMask.Length]));
        var maskedNegLogProbs = targetLogProbs.Multiply(mask).Negate();

        var summed = maskedNegLogProbs.Sum(axis: 0, keepDims: true);
        var divisor = new Variable(TensorValue.FromValues([maskedCount], [1]));
        return summed.Divide(divisor);
    }

    /// <summary>
    /// log_softmax(x) evaluated at each position's actual target token -
    /// the shared core of <see cref="Compute"/> and <see cref="ComputeMasked"/>,
    /// which differ only in which positions get averaged over.
    /// log_softmax(x) = x - logsumexp(x), computed this way (rather than
    /// Softmax().Log()) so the normalising constant is worked out in
    /// log-space directly instead of round-tripping through a softmax
    /// value that could already have underflowed to 0. logsumexp(x) is
    /// computed via the standard shifted form - max(x) + log(sum(exp(x -
    /// max(x)))) - so every Exp argument stays <= 0 (TASK-017's "safe
    /// softmax" trick), avoiding overflow for large-magnitude logits.
    /// maxValue is wrapped as a constant (no parent op), which
    /// deliberately stops gradient flowing through the max itself -
    /// standard practice, and correct here since d(logsumexp)/dx_i =
    /// softmax(x)_i regardless of the shift.
    /// </summary>
    private static Variable TargetLogProbs(Variable logits, int[] targetTokenIds)
    {
        int axis = logits.Value.Shape.Length - 1;

        var maxValue = new Variable(logits.Value.Max(axis, keepDims: true));
        var shifted = logits.Subtract(maxValue);
        var logSumExpShifted = shifted.Exp().Sum(axis, keepDims: true).Log();
        var logSoftmax = shifted.Subtract(logSumExpShifted);

        return logSoftmax.GatherColumns(targetTokenIds);
    }
}
