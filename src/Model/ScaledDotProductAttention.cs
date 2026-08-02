using Tensor;
using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// The core attention computation, with no learned parameters of its own -
/// <see cref="MultiHeadAttention"/> projects into Q/K/V and calls this per
/// head. Operates on any leading batch/head dimensions (e.g. [numHeads,
/// seqLen, headDim]): only the last two dimensions are the actual
/// sequence/feature dims that matmul and softmax act over.
/// </summary>
public static class ScaledDotProductAttention
{
    /// <summary>
    /// Attention(Q,K,V) = softmax(Q @ K^T / sqrt(headDim)) @ V.
    /// </summary>
    /// <param name="causal">
    /// When true, masks out attention to future positions (position i may
    /// only attend to positions 0..i) - required for an autoregressive,
    /// decoder-only model: without it, next-token prediction during
    /// training would let the model "see" the very token it's supposed to
    /// predict.
    /// </param>
    public static Variable Compute(Variable query, Variable key, Variable value, bool causal)
    {
        int headDim = query.Value.Shape[^1];
        int seqLen = query.Value.Shape[^2];

        var keyTransposed = key.Transpose(key.Value.Shape.Length - 2, key.Value.Shape.Length - 1);
        var scale = new Variable(TensorValue.FromValues([1f / MathF.Sqrt(headDim)], [1]));
        var scores = query.MatMul(keyTransposed).Multiply(scale);

        if (causal)
        {
            scores = scores.Add(CausalMask(seqLen));
        }

        var weights = scores.Softmax(axis: scores.Value.Shape.Length - 1);
        return weights.MatMul(value);
    }

    /// <summary>
    /// A [seqLen, seqLen] additive mask: 0 where position i may attend to
    /// position j (j &lt;= i), -infinity where it may not (j &gt; i).
    /// -infinity rather than a large-but-finite negative number so
    /// softmax's exp() zeroes those entries out exactly, with no residual
    /// (however small) attention leaking into the future.
    /// </summary>
    private static Variable CausalMask(int seqLen)
    {
        var values = new float[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                values[i * seqLen + j] = j > i ? float.NegativeInfinity : 0f;
            }
        }
        return new Variable(TensorValue.FromValues(values, [seqLen, seqLen]));
    }
}
