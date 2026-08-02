using Tensor;
using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// A learned token-id -> dense-vector lookup table: the first layer of a
/// transformer, turning discrete token ids (from the tokeniser's
/// vocabulary) into vectors the rest of the model can do math on. Backed
/// by a trainable <see cref="Variable"/> so gradients flow back into the
/// embedding table during training.
/// </summary>
public sealed class TokenEmbedding
{
    public int VocabSize { get; }
    public int EmbeddingDim { get; }
    public Variable Weight { get; }

    public TokenEmbedding(int vocabSize, int embeddingDim, Random? random = null)
    {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        Weight = new Variable(InitialWeights(vocabSize, embeddingDim, random ?? new Random()));
    }

    /// <summary>Looks up a sequence of token ids, returning a [sequenceLength, embeddingDim] variable.</summary>
    public Variable Forward(int[] tokenIds) => Weight.GatherRows(tokenIds);

    /// <summary>
    /// Small random values (mean 0, std 0.02 - the same scale GPT-2 uses
    /// for its embedding init) rather than zeros: identical rows would
    /// otherwise get identical gradients forever and never differentiate.
    /// </summary>
    private static TensorValue InitialWeights(int vocabSize, int embeddingDim, Random random)
    {
        const float stdDev = 0.02f;
        var values = new float[vocabSize * embeddingDim];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = SampleGaussian(random) * stdDev;
        }
        return TensorValue.FromValues(values, [vocabSize, embeddingDim]);
    }

    /// <summary>
    /// Standard normal sample via the Box-Muller transform - .NET's Random
    /// only gives uniform samples, and this is a from-first-principles
    /// project, so no external distribution library either.
    /// </summary>
    internal static float SampleGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}
