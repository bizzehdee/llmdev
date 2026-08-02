using Tensor;

namespace Model;

/// <summary>
/// A decoder-only, GPT-2-style language model: token + positional
/// embeddings, a stack of causal <see cref="TransformerBlock"/>s, a final
/// layernorm, and an output projection back to logits over the
/// tokeniser's vocabulary. The output projection reuses (rather than
/// duplicates) <see cref="Model.TokenEmbedding"/>'s weight matrix,
/// transposed - "weight tying" - which is what GPT-2 itself does: it
/// halves the parameter count of what would otherwise be two separate
/// [vocabSize, embeddingDim]-sized matrices, and is a reasonable prior
/// besides (a token's input representation and its output "am I likely
/// next" score plausibly share structure).
/// </summary>
public sealed class GptModel
{
    public int VocabSize { get; }
    public int EmbeddingDim { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int MaxSequenceLength { get; }

    public TokenEmbedding TokenEmbedding { get; }
    public PositionalEmbedding PositionalEmbedding { get; }
    public IReadOnlyList<TransformerBlock> Blocks { get; }
    public LayerNorm FinalNorm { get; }

    public GptModel(int vocabSize, int embeddingDim, int numLayers, int numHeads, int maxSequenceLength, int? feedForwardHiddenDim = null, Random? random = null)
    {
        VocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        NumLayers = numLayers;
        NumHeads = numHeads;
        MaxSequenceLength = maxSequenceLength;

        random ??= new Random();
        TokenEmbedding = new TokenEmbedding(vocabSize, embeddingDim, random);
        PositionalEmbedding = new PositionalEmbedding(maxSequenceLength, embeddingDim, random);

        var blocks = new List<TransformerBlock>(numLayers);
        for (int i = 0; i < numLayers; i++)
        {
            blocks.Add(new TransformerBlock(embeddingDim, numHeads, feedForwardHiddenDim, causal: true, random));
        }
        Blocks = blocks;

        FinalNorm = new LayerNorm(embeddingDim);
    }

    /// <summary>tokenIds (length &lt;= <see cref="MaxSequenceLength"/>) -> logits, shape [sequenceLength, vocabSize].</summary>
    public Variable Forward(int[] tokenIds)
    {
        var x = TokenEmbedding.Forward(tokenIds).Add(PositionalEmbedding.Forward(tokenIds.Length));

        foreach (var block in Blocks)
        {
            x = block.Forward(x);
        }

        x = FinalNorm.Forward(x);

        var tiedOutputWeight = TokenEmbedding.Weight.Transpose(0, 1); // [embeddingDim, vocabSize]
        return x.MatMul(tiedOutputWeight);
    }

    /// <summary>
    /// Every trainable parameter, in a fixed deterministic order (used by
    /// both an optimizer and by checkpointing to line up saved values with
    /// the right Variable). Excludes the output projection: it's weight-tied
    /// to TokenEmbedding.Weight (see the class doc comment), not a separate
    /// parameter, so including it again here would double-count and double
    /// its effective learning rate.
    /// </summary>
    public IReadOnlyList<Variable> Parameters() =>
        [.. TokenEmbedding.Parameters(), .. PositionalEmbedding.Parameters(), .. Blocks.SelectMany(b => b.Parameters()), .. FinalNorm.Parameters()];
}
