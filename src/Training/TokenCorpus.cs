using Common;
using Tokeniser;

namespace Training;

/// <summary>
/// A tokenised corpus - a flat stream of token ids - held in a disk-backed
/// <see cref="MappedArray{T}"/> rather than the managed heap, per PLAN.md's
/// memory constraint: a training corpus can be large, and <see cref="BatchSampler"/>
/// only ever needs to read small fixed-length windows out of it, not hold
/// the whole thing in RAM.
///
/// Getting the token ids into an <see cref="IReadOnlyList{T}"/> in the
/// first place is the caller's problem for the general constructor - most
/// naturally by calling <c>BpeTokeniser.Encode</c> on some text, though
/// that's a poor fit for a *very* large corpus (`Encode` is tuned for
/// short/moderate text, the same simple merge-scan approach TASK-001
/// explicitly avoided for *training* the tokeniser on large corpora). The
/// <see cref="EncodedCorpus"/> constructor below is the resolution TASK-018
/// (`BpeTokeniser.EncodeBulk`) was built for: it streams token ids
/// straight from one disk-backed source into another, never materialising
/// the whole corpus as a single in-memory collection.
/// </summary>
public sealed class TokenCorpus : IDisposable
{
    private readonly MappedArray<int> _tokens;

    public int Length { get; }

    public TokenCorpus(IReadOnlyList<int> tokenIds, string scratchDirectory)
    {
        Length = tokenIds.Count;
        _tokens = new MappedArray<int>(Math.Max(Length, 1), scratchDirectory);
        for (int i = 0; i < Length; i++)
        {
            _tokens[i] = tokenIds[i];
        }
    }

    /// <summary>
    /// Builds a corpus directly from a bulk-encoded source (TASK-018's
    /// <c>BpeTokeniser.EncodeBulk</c>) - the large-corpus path, streaming
    /// one token at a time from <paramref name="encoded"/>'s own
    /// disk-backed storage into this corpus's, rather than round-tripping
    /// through a managed-heap <see cref="IReadOnlyList{T}"/> in between.
    /// </summary>
    public TokenCorpus(EncodedCorpus encoded, string scratchDirectory)
    {
        Length = encoded.Length;
        _tokens = new MappedArray<int>(Math.Max(Length, 1), scratchDirectory);
        for (int i = 0; i < Length; i++)
        {
            _tokens[i] = encoded[i];
        }
    }

    public int this[int index] => _tokens[index];

    public void Dispose() => _tokens.Dispose();
}
