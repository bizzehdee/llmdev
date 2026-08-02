namespace Training;

/// <summary>
/// Draws fixed-length next-token-prediction windows from a
/// <see cref="TokenCorpus"/>. Each example needs contextLength+1 tokens
/// (the input window plus one more token to shift into the target), so a
/// window starting at position i is Input = corpus[i..i+contextLength-1],
/// Target = corpus[i+1..i+contextLength].
///
/// The model this feeds (<c>GptModel</c>, TASK-009) has no batch dimension
/// of its own - it processes one sequence at a time - so "batch" here just
/// means "a set of examples to accumulate gradients over before an
/// optimizer step" (TASK-012's job), not a single batched tensor
/// computation.
/// </summary>
public sealed class BatchSampler
{
    private readonly TokenCorpus _corpus;
    private readonly int _contextLength;
    private readonly Random _random;

    public BatchSampler(TokenCorpus corpus, int contextLength, Random? random = null)
    {
        if (contextLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(contextLength), "Context length must be at least 1.");
        }
        if (corpus.Length < contextLength + 1)
        {
            throw new ArgumentException($"Corpus has {corpus.Length} tokens, but needs at least {contextLength + 1} (context length {contextLength} plus one for the target shift).");
        }

        _corpus = corpus;
        _contextLength = contextLength;
        _random = random ?? new Random();
    }

    /// <summary>The last valid start index for a window (inclusive).</summary>
    public int MaxStartIndex => _corpus.Length - _contextLength - 1;

    public TrainingExample GetExample(int startIndex)
    {
        if (startIndex < 0 || startIndex > MaxStartIndex)
        {
            throw new ArgumentOutOfRangeException(nameof(startIndex), $"Start index must be in [0,{MaxStartIndex}].");
        }

        var input = new int[_contextLength];
        var target = new int[_contextLength];
        for (int i = 0; i < _contextLength; i++)
        {
            input[i] = _corpus[startIndex + i];
            target[i] = _corpus[startIndex + i + 1];
        }
        return new TrainingExample(input, target);
    }

    /// <summary>Draws <paramref name="batchSize"/> examples at independently random start positions.</summary>
    public TrainingExample[] SampleBatch(int batchSize)
    {
        var batch = new TrainingExample[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            batch[i] = GetExample(_random.Next(0, MaxStartIndex + 1));
        }
        return batch;
    }
}
