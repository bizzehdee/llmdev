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
    /// <param name="queryOffset">
    /// TASK-020's KV-cached generation path calls Q, at a given step, over
    /// only the *new* tokens, while K/V cover the *whole* cached sequence
    /// so far - so query row i's real absolute position is
    /// <paramref name="queryOffset"/> + i, not i, and that's what the
    /// causal mask must compare against key position j. Zero (the
    /// default) reproduces the ordinary square mask used everywhere else,
    /// where query and key cover the same, single sequence.
    /// </param>
    public static Variable Compute(Variable query, Variable key, Variable value, bool causal, int queryOffset = 0)
    {
        int headDim = query.Value.Shape[^1];
        int queryLen = query.Value.Shape[^2];
        int keyLen = key.Value.Shape[^2];

        var keyTransposed = key.Transpose(key.Value.Shape.Length - 2, key.Value.Shape.Length - 1);
        var scale = new Variable(TensorValue.FromValues([1f / MathF.Sqrt(headDim)], [1]));
        var scores = query.MatMul(keyTransposed).Multiply(scale);

        if (causal)
        {
            scores = scores.Add(CausalMask(queryLen, keyLen, queryOffset));
        }

        var weights = scores.Softmax(axis: scores.Value.Shape.Length - 1);
        return weights.MatMul(value);
    }

    /// <summary>
    /// A [queryLen, keyLen] additive mask: 0 where query row i (absolute
    /// position queryOffset + i) may attend to key position j (j &lt;=
    /// queryOffset + i), -infinity where it may not. -infinity rather than
    /// a large-but-finite negative number so softmax's exp() zeroes those
    /// entries out exactly, with no residual (however small) attention
    /// leaking into the future.
    /// </summary>
    private static Variable CausalMask(int queryLen, int keyLen, int queryOffset)
    {
        var values = new float[queryLen * keyLen];
        for (int i = 0; i < queryLen; i++)
        {
            for (int j = 0; j < keyLen; j++)
            {
                values[i * keyLen + j] = j > queryOffset + i ? float.NegativeInfinity : 0f;
            }
        }
        return new Variable(TensorValue.FromValues(values, [queryLen, keyLen]));
    }
}
