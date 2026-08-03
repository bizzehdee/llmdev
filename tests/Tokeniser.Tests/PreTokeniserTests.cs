using Tokeniser;
using Xunit;

namespace Tokeniser.Tests;

public class PreTokeniserTests
{
    [Fact]
    public void Split_WordsGetALeadingSpaceGroupedWithThem()
    {
        // cl100k_base-style: a run of letters absorbs one leading
        // non-letter/non-digit character (typically a space) into the
        // same chunk, rather than emitting the space as its own chunk.
        var chunks = PreTokeniser.Split("hello world").ToList();

        Assert.Equal(new[] { "hello", " world" }, chunks);
    }

    [Fact]
    public void Split_DigitRunsAreCappedAtThreeCharacters()
    {
        var chunks = PreTokeniser.Split("12345").ToList();

        Assert.Equal(new[] { "123", "45" }, chunks);
    }

    [Fact]
    public void Split_ContractionsAreSplitFromTheirStem()
    {
        var chunks = PreTokeniser.Split("don't").ToList();

        Assert.Equal(new[] { "don", "'t" }, chunks);
    }

    [Fact]
    public void Split_PunctuationRunsStayTogether()
    {
        var chunks = PreTokeniser.Split("wait...!").ToList();

        Assert.Equal(new[] { "wait", "...!" }, chunks);
    }

    [Theory]
    [InlineData("hello world")]
    [InlineData("don't stop 12345 times!!")]
    [InlineData("  leading and trailing whitespace   ")]
    [InlineData("emoji test \U0001F600 done")]
    [InlineData("CJK text: 你好世界")]
    [InlineData("")]
    [InlineData("\n\n\nmultiple newlines\n\n\n")]
    public void Split_ConcatenatedChunksReproduceTheOriginalTextExactly(string text)
    {
        // The correctness bar for chunking itself, independent of BPE:
        // no character may be gained, lost, or reordered.
        var chunks = PreTokeniser.Split(text).ToList();

        Assert.Equal(text, string.Concat(chunks));
    }

    // TASK-029: the streaming TextReader overload must chunk identically to
    // the in-memory string overload, however small the read buffer - the
    // whole point is that a chunk spanning a block boundary is still
    // returned whole, never split.

    [Theory]
    [InlineData("hello world")]
    [InlineData("don't stop 12345 times!!")]
    [InlineData("  leading and trailing whitespace   ")]
    [InlineData("emoji test \U0001F600 done")]
    [InlineData("CJK text: 你好世界")]
    [InlineData("")]
    [InlineData("\n\n\nmultiple newlines\n\n\n")]
    [InlineData("a repeated word word word word word word word word word word end")]
    public void Split_StreamOverload_MatchesInMemoryOverload_AcrossVariousBufferSizes(string text)
    {
        var expected = PreTokeniser.Split(text).ToList();

        foreach (int bufferSize in new[] { 1, 2, 3, 5, 64 })
        {
            using var reader = new StringReader(text);
            var actual = PreTokeniser.Split(reader, bufferSize).ToList();

            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Split_StreamOverload_WordSpanningManyTinyBlocksStaysOneChunk()
    {
        string text = "supercalifragilisticexpialidocious and more";

        using var reader = new StringReader(text);
        var chunks = PreTokeniser.Split(reader, bufferSize: 1).ToList();

        Assert.Contains("supercalifragilisticexpialidocious", chunks);
        Assert.Equal(text, string.Concat(chunks));
    }

    [Fact]
    public void Split_StreamOverload_EmptyReaderProducesNoChunks()
    {
        using var reader = new StringReader("");

        var chunks = PreTokeniser.Split(reader).ToList();

        Assert.Empty(chunks);
    }
}
