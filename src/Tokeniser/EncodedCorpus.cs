using Common;

namespace Tokeniser;

/// <summary>
/// Result of <see cref="BpeTokeniser.EncodeBulk"/>: a disk-backed sequence
/// of token ids covering an entire corpus (TASK-018), so a large corpus's
/// encoded form never needs to fit on the managed heap. The backing
/// <see cref="MappedArray{T}"/> is over-allocated to the input's raw byte
/// count (an upper bound - merges only ever remove tokens); only indices
/// [0, Length) hold meaningful data.
/// </summary>
public sealed class EncodedCorpus : IDisposable
{
    private readonly MappedArray<int> _tokens;

    public int Length { get; }

    internal EncodedCorpus(MappedArray<int> tokens, int length)
    {
        _tokens = tokens;
        Length = length;
    }

    public int this[int index]
    {
        get
        {
            if (index < 0 || index >= Length)
            {
                throw new IndexOutOfRangeException($"Index {index} out of range for length {Length}.");
            }
            return _tokens[index];
        }
    }

    public void Dispose() => _tokens.Dispose();
}
