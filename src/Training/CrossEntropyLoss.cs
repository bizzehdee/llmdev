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

        // log_softmax(x) = x - log(sum(exp(x))), computed this way (rather
        // than Softmax().Log()) so the normalising constant is worked out
        // in log-space directly instead of round-tripping through a
        // softmax value that could already have underflowed to 0. Still no
        // max-subtraction stability trick (see Variable.Softmax) - same
        // caveat applies here.
        var logSumExp = logits.Exp().Sum(axis, keepDims: true).Log();
        var logSoftmax = logits.Subtract(logSumExp);

        var targetLogProbs = logSoftmax.GatherColumns(targetTokenIds);
        return targetLogProbs.Negate().Mean(axis: 0, keepDims: true);
    }
}
