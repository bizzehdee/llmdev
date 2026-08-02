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
    private const float InitStdDev = 0.02f; // GPT-2's embedding init scale

    public int VocabSize { get; }
    public int EmbeddingDim { get; }
    public Variable Weight { get; }

    public TokenEmbedding(int vocabSize, int embeddingDim, Random? random = null)
    {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        Weight = new Variable(GaussianInit.Matrix(vocabSize, embeddingDim, InitStdDev, random ?? new Random()));
    }

    /// <summary>Looks up a sequence of token ids, returning a [sequenceLength, embeddingDim] variable.</summary>
    public Variable Forward(int[] tokenIds) => Weight.GatherRows(tokenIds);
}
