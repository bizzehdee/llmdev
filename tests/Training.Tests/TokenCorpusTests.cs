using Training;
using Xunit;

namespace Training.Tests;

public class TokenCorpusTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "training-tests-scratch");

    static TokenCorpusTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void Constructor_ExposesLength()
    {
        using var corpus = new TokenCorpus([1, 2, 3, 4, 5], ScratchDirectory);

        Assert.Equal(5, corpus.Length);
    }

    [Fact]
    public void Indexer_ReturnsOriginalTokenIds()
    {
        using var corpus = new TokenCorpus([10, 20, 30], ScratchDirectory);

        Assert.Equal(10, corpus[0]);
        Assert.Equal(20, corpus[1]);
        Assert.Equal(30, corpus[2]);
    }

    [Fact]
    public void Constructor_EmptyCorpusHasZeroLength()
    {
        using var corpus = new TokenCorpus([], ScratchDirectory);

        Assert.Equal(0, corpus.Length);
    }
}
