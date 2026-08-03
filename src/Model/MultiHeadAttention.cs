using Tensor;

namespace Model;

/// <summary>
/// Multi-head self-attention: projects the input into Q/K/V, splits each
/// into <see cref="NumHeads"/> independent heads so different heads can
/// learn to attend to different kinds of relationships, runs
/// <see cref="ScaledDotProductAttention"/> per head in parallel (as a batch
/// dimension), concatenates the heads back together, and projects the
/// result once more.
/// </summary>
public sealed class MultiHeadAttention
{
    private const float InitStdDev = 0.02f; // GPT-2's init scale

    public int EmbeddingDim { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public bool Causal { get; }

    public Variable QueryWeight { get; }
    public Variable KeyWeight { get; }
    public Variable ValueWeight { get; }
    public Variable OutputWeight { get; }

    public MultiHeadAttention(int embeddingDim, int numHeads, bool causal = true, Random? random = null)
    {
        if (embeddingDim % numHeads != 0)
        {
            throw new ArgumentException($"embeddingDim ({embeddingDim}) must be divisible by numHeads ({numHeads}).");
        }

        EmbeddingDim = embeddingDim;
        NumHeads = numHeads;
        HeadDim = embeddingDim / numHeads;
        Causal = causal;

        random ??= new Random();
        QueryWeight = new Variable(GaussianInit.Matrix(embeddingDim, embeddingDim, InitStdDev, random));
        KeyWeight = new Variable(GaussianInit.Matrix(embeddingDim, embeddingDim, InitStdDev, random));
        ValueWeight = new Variable(GaussianInit.Matrix(embeddingDim, embeddingDim, InitStdDev, random));
        OutputWeight = new Variable(GaussianInit.Matrix(embeddingDim, embeddingDim, InitStdDev, random));
    }

    /// <summary>
    /// <paramref name="input"/> is [sequenceLength, embeddingDim]; the
    /// result is the same shape.
    /// </summary>
    public Variable Forward(Variable input)
    {
        int seqLen = input.Value.Shape[0];

        var query = SplitHeads(input.MatMul(QueryWeight), seqLen);
        var key = SplitHeads(input.MatMul(KeyWeight), seqLen);
        var value = SplitHeads(input.MatMul(ValueWeight), seqLen);

        var attended = ScaledDotProductAttention.Compute(query, key, value, Causal);

        var merged = CombineHeads(attended, seqLen);
        return merged.MatMul(OutputWeight);
    }

    /// <summary>
    /// TASK-020's KV-cached step: <paramref name="input"/> is only the
    /// *new* tokens ([newLen, embeddingDim], newLen == 1 in the common
    /// steady-state generation case, but any length works - e.g. an
    /// initial multi-token prompt). Q/K/V are computed for just those new
    /// positions; the new K/V are appended onto <paramref name="cache"/>'s
    /// existing ones for this layer (see <see cref="Tensor.Tensor.Concat"/>),
    /// and attention runs the new Q against the *whole* resulting K/V, so
    /// the new tokens see every position processed so far without any of
    /// those earlier positions being recomputed.
    /// </summary>
    public Variable ForwardIncremental(Variable input, GenerationCache cache, int layerIndex)
    {
        int newLen = input.Value.Shape[0];
        int queryOffset = cache.Length;

        var query = SplitHeads(input.MatMul(QueryWeight), newLen);
        var newKey = SplitHeads(input.MatMul(KeyWeight), newLen);
        var newValue = SplitHeads(input.MatMul(ValueWeight), newLen);

        var cachedKey = cache.GetKey(layerIndex);
        var cachedValue = cache.GetValue(layerIndex);
        var fullKeyValue = cachedKey is null ? newKey.Value : cachedKey.Concat(newKey.Value, axis: 1);
        var fullValueValue = cachedValue is null ? newValue.Value : cachedValue.Concat(newValue.Value, axis: 1);
        cache.SetLayer(layerIndex, fullKeyValue, fullValueValue);

        var attended = ScaledDotProductAttention.Compute(query, new Variable(fullKeyValue), new Variable(fullValueValue), Causal, queryOffset);

        var merged = CombineHeads(attended, newLen);
        return merged.MatMul(OutputWeight);
    }

    public IReadOnlyList<Variable> Parameters() => [QueryWeight, KeyWeight, ValueWeight, OutputWeight];

    /// <summary>[seqLen, embeddingDim] -> [numHeads, seqLen, headDim], so batched matmul treats heads as independent batches.</summary>
    private Variable SplitHeads(Variable x, int seqLen) =>
        x.Reshape([seqLen, NumHeads, HeadDim]).Transpose(0, 1);

    /// <summary>
    /// [numHeads, seqLen, headDim] -> [seqLen, embeddingDim], the inverse of
    /// <see cref="SplitHeads"/>. Transpose-then-reshape is only safe because
    /// this project's Tensor always materialises a genuinely contiguous
    /// buffer for Transpose (see Tensor.cs) rather than a strided view - the
    /// classic version of this op elsewhere requires an explicit
    /// ".contiguous()" call in between for exactly that reason.
    /// </summary>
    private Variable CombineHeads(Variable x, int seqLen) =>
        x.Transpose(0, 1).Reshape([seqLen, EmbeddingDim]);
}
