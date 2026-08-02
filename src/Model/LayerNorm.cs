using Tensor;
using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// Normalises each position's feature vector (the last axis) to zero mean
/// and unit variance, then applies a learned per-feature scale and shift.
/// Used before attention and before the feed-forward layer in
/// <see cref="TransformerBlock"/> (GPT-2's "pre-norm" placement) to keep
/// activations at a consistent scale through a deep stack of blocks -
/// without it, residual connections tend to make activation magnitudes
/// grow with depth and training becomes unstable.
/// </summary>
public sealed class LayerNorm
{
    private const float Epsilon = 1e-5f;

    public int EmbeddingDim { get; }
    public Variable Gamma { get; }
    public Variable Beta { get; }

    public LayerNorm(int embeddingDim)
    {
        EmbeddingDim = embeddingDim;
        Gamma = new Variable(TensorValue.FromValues(Enumerable.Repeat(1f, embeddingDim).ToArray(), [embeddingDim]));
        Beta = new Variable(TensorValue.Zeros([embeddingDim]));
    }

    public Variable Forward(Variable x)
    {
        int axis = x.Value.Shape.Length - 1;

        var mean = x.Mean(axis, keepDims: true);
        var centered = x.Subtract(mean);
        var variance = centered.Multiply(centered).Mean(axis, keepDims: true);

        var epsilon = new Variable(TensorValue.FromValues([Epsilon], [1]));
        var standardDeviation = variance.Add(epsilon).Sqrt();
        var normalized = centered.Divide(standardDeviation);

        return normalized.Multiply(Gamma).Add(Beta);
    }
}
