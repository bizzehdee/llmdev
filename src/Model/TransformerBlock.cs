using Tensor;

namespace Model;

/// <summary>
/// One decoder-only transformer block, GPT-2-style "pre-norm" layout:
/// normalise, attend, add the residual; normalise, feed-forward, add the
/// residual. The residual connections (adding the block's input back onto
/// its output at each stage, rather than replacing it) are what let
/// gradients flow through many stacked blocks without vanishing, and are
/// why every sub-layer here preserves the [sequenceLength, embeddingDim]
/// shape end to end.
/// </summary>
public sealed class TransformerBlock
{
    public int EmbeddingDim { get; }
    public MultiHeadAttention Attention { get; }
    public FeedForward FeedForward { get; }
    public LayerNorm PreAttentionNorm { get; }
    public LayerNorm PreFeedForwardNorm { get; }

    public TransformerBlock(int embeddingDim, int numHeads, int? feedForwardHiddenDim = null, bool causal = true, Random? random = null)
    {
        EmbeddingDim = embeddingDim;
        random ??= new Random();

        Attention = new MultiHeadAttention(embeddingDim, numHeads, causal, random);
        FeedForward = new FeedForward(embeddingDim, feedForwardHiddenDim ?? embeddingDim * 4, random);
        PreAttentionNorm = new LayerNorm(embeddingDim);
        PreFeedForwardNorm = new LayerNorm(embeddingDim);
    }

    /// <summary>[sequenceLength, embeddingDim] in, same shape out.</summary>
    public Variable Forward(Variable x)
    {
        var attended = Attention.Forward(PreAttentionNorm.Forward(x));
        x = x.Add(attended);

        var fed = FeedForward.Forward(PreFeedForwardNorm.Forward(x));
        x = x.Add(fed);

        return x;
    }

    public IReadOnlyList<Variable> Parameters() =>
        [.. Attention.Parameters(), .. FeedForward.Parameters(), .. PreAttentionNorm.Parameters(), .. PreFeedForwardNorm.Parameters()];
}
