using Tensor;

namespace Model;

/// <summary>
/// A learned position -> dense-vector lookup table. Attention has no
/// inherent sense of sequence order (it's a weighted sum over all
/// positions, symmetric in position by default), so this is added to the
/// token embedding to give the model something to tell "first token" from
/// "fifth token" with. Learned rather than sinusoidal/RoPE for this first
/// pass, per PLAN.md - fewer moving parts, at the cost of not
/// generalising to sequence lengths longer than <see cref="MaxSequenceLength"/>
/// seen during training.
/// </summary>
public sealed class PositionalEmbedding
{
    private const float InitStdDev = 0.02f; // GPT-2's embedding init scale

    public int MaxSequenceLength { get; }
    public int EmbeddingDim { get; }
    public Variable Weight { get; }

    public PositionalEmbedding(int maxSequenceLength, int embeddingDim, Random? random = null)
    {
        MaxSequenceLength = maxSequenceLength;
        EmbeddingDim = embeddingDim;
        Weight = new Variable(GaussianInit.Matrix(maxSequenceLength, embeddingDim, InitStdDev, random ?? new Random()));
    }

    /// <summary>
    /// Returns the positional embedding for positions 0..sequenceLength-1,
    /// as a [sequenceLength, embeddingDim] variable - shape-compatible to
    /// add directly to a <see cref="TokenEmbedding"/> lookup over the same
    /// sequence.
    /// </summary>
    public Variable Forward(int sequenceLength)
    {
        if (sequenceLength > MaxSequenceLength)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), $"Sequence length {sequenceLength} exceeds the {MaxSequenceLength} positions this embedding was sized for.");
        }

        var positionIds = new int[sequenceLength];
        for (int i = 0; i < sequenceLength; i++)
        {
            positionIds[i] = i;
        }
        return Weight.GatherRows(positionIds);
    }
}
