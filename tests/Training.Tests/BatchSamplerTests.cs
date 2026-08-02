using Training;
using Xunit;

namespace Training.Tests;

public class BatchSamplerTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "training-tests-scratch");

    static BatchSamplerTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void Constructor_ContextLengthLessThanOneThrows()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);

        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchSampler(corpus, contextLength: 0));
    }

    [Fact]
    public void Constructor_CorpusTooShortForContextLengthThrows()
    {
        using var corpus = new TokenCorpus([1, 2, 3], ScratchDirectory);

        // Needs contextLength+1 = 4 tokens, corpus only has 3.
        Assert.Throws<ArgumentException>(() => new BatchSampler(corpus, contextLength: 3));
    }

    [Fact]
    public void Constructor_ExactlyEnoughTokensSucceeds()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4], ScratchDirectory);

        var sampler = new BatchSampler(corpus, contextLength: 3);

        Assert.Equal(0, sampler.MaxStartIndex);
    }

    [Fact]
    public void MaxStartIndex_ComputedFromCorpusLengthAndContextLength()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5, 6, 7], ScratchDirectory); // length 7
        var sampler = new BatchSampler(corpus, contextLength: 3);

        // Need contextLength+1=4 tokens per window; last valid start is 7-4=3.
        Assert.Equal(3, sampler.MaxStartIndex);
    }

    [Fact]
    public void GetExample_TargetIsInputShiftedByOne()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 3);

        var example = sampler.GetExample(startIndex: 0);

        Assert.Equal(new[] { 1, 2, 3 }, example.Input);
        Assert.Equal(new[] { 2, 3, 4 }, example.Target);
    }

    [Fact]
    public void GetExample_AtDifferentStartIndexShiftsWindow()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 3);

        var example = sampler.GetExample(startIndex: 1);

        Assert.Equal(new[] { 2, 3, 4 }, example.Input);
        Assert.Equal(new[] { 3, 4, 5 }, example.Target);
    }

    [Fact]
    public void GetExample_NegativeStartIndexThrows()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.GetExample(-1));
    }

    [Fact]
    public void GetExample_StartIndexBeyondMaxThrows()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.GetExample(sampler.MaxStartIndex + 1));
    }

    [Fact]
    public void SampleBatch_ReturnsRequestedNumberOfExamples()
    {
        using var corpus = new TokenCorpus(Enumerable.Range(0, 100).ToArray(), ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 8, new Random(1));

        var batch = sampler.SampleBatch(batchSize: 16);

        Assert.Equal(16, batch.Length);
    }

    [Fact]
    public void SampleBatch_EveryExampleIsAValidWindowOfTheCorpus()
    {
        var tokens = Enumerable.Range(0, 50).ToArray();
        using var corpus = new TokenCorpus(tokens, ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 5, new Random(2));

        var batch = sampler.SampleBatch(batchSize: 20);

        foreach (var example in batch)
        {
            Assert.Equal(5, example.Input.Length);
            Assert.Equal(5, example.Target.Length);
            // Since the corpus is just 0,1,2,...,49, every valid window is
            // consecutive integers, and target = input + 1 elementwise.
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(example.Input[i] + 1, example.Target[i]);
            }
            for (int i = 1; i < 5; i++)
            {
                Assert.Equal(example.Input[i - 1] + 1, example.Input[i]);
            }
        }
    }

    [Fact]
    public void SampleBatch_SameSeedProducesSameBatch()
    {
        var tokens = Enumerable.Range(0, 50).ToArray();
        using var corpusA = new TokenCorpus(tokens, ScratchDirectory);
        using var corpusB = new TokenCorpus(tokens, ScratchDirectory);
        var samplerA = new BatchSampler(corpusA, contextLength: 5, new Random(42));
        var samplerB = new BatchSampler(corpusB, contextLength: 5, new Random(42));

        var batchA = samplerA.SampleBatch(10);
        var batchB = samplerB.SampleBatch(10);

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(batchA[i].Input, batchB[i].Input);
            Assert.Equal(batchA[i].Target, batchB[i].Target);
        }
    }
}
