using Common;

namespace Training;

/// <summary>
/// A tokenised corpus - a flat stream of token ids - held in a disk-backed
/// <see cref="MappedArray{T}"/> rather than the managed heap, per PLAN.md's
/// memory constraint: a training corpus can be large, and <see cref="BatchSampler"/>
/// only ever needs to read small fixed-length windows out of it, not hold
/// the whole thing in RAM.
///
/// Getting the token ids into an <see cref="IReadOnlyList{T}"/> in the
/// first place is the caller's problem - most naturally by calling
/// <c>BpeTokeniser.Encode</c> on some text. That's a real limitation for a
/// *very* large corpus: `Encode` is tuned for short/moderate text (the
/// same simple merge-scan approach TASK-001 explicitly avoided for
/// *training* the tokeniser on large corpora, for exactly the performance
/// reason described in BpeTokeniser.cs), not efficient bulk encoding of
/// hundreds of MB of text. That's out of scope for batching itself and
/// noted as a follow-up in TASK.md if it turns out to matter once an
/// actual training run is attempted.
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

    public int this[int index] => _tokens[index];

    public void Dispose() => _tokens.Dispose();
}
