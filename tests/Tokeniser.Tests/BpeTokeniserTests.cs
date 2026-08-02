using Tokeniser;
using Xunit;

namespace Tokeniser.Tests;

public class BpeTokeniserTests
{
    // Test corpora are tiny, so it doesn't matter that this may land on tmpfs
    // (unlike the real CLI, which insists on genuine disk - see Program.cs).
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "tokeniser-tests-scratch");

    static BpeTokeniserTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void NewTokeniser_HasBaseByteVocabulary()
    {
        var tokeniser = new BpeTokeniser();

        Assert.Equal(256, tokeniser.VocabSize);
    }

    [Fact]
    public void Train_LearnsMergesAndGrowsVocabulary()
    {
        var path = WriteTempFile("ababababab ababababab ababababab");
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { path }, targetVocabSize: 260, scratchDirectory: ScratchDirectory);

            Assert.True(tokeniser.VocabSize > 256);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Train_StopsAtTargetVocabSize()
    {
        var path = WriteTempFile(string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 50)));
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { path }, targetVocabSize: 300, scratchDirectory: ScratchDirectory);

            Assert.True(tokeniser.VocabSize <= 300);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData("hello world")]
    [InlineData("The quick brown fox jumps over the lazy dog.")]
    [InlineData("")]
    [InlineData("unicode: café 😀 你好")]
    public void EncodeThenDecode_RoundtripsForUntrainedTokeniser(string text)
    {
        var tokeniser = new BpeTokeniser();

        var encoded = tokeniser.Encode(text);
        var decoded = tokeniser.Decode(encoded);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void EncodeThenDecode_RoundtripsAfterTraining()
    {
        var corpus = "the quick brown fox jumps over the lazy dog. the dog barks. the fox runs.";
        var path = WriteTempFile(corpus);
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { path }, targetVocabSize: 280, scratchDirectory: ScratchDirectory);

            var encoded = tokeniser.Encode(corpus);
            var decoded = tokeniser.Decode(encoded);

            Assert.Equal(corpus, decoded);
            Assert.True(encoded.Count < corpus.Length, "Trained encoding should compress below one token per character.");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData("aaaaaaaaaaaaaaaa")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")]
    [InlineData("abababababababababababababababababababababababababababababababababababababababab")]
    public void Train_HandlesOverlappingRepeatedPairsCorrectly(string corpus)
    {
        // Repeated/overlapping runs of the same pair (e.g. "aaaa") are the case where an
        // incremental, position-based merge implementation is most likely to double-count
        // or skip a merge, since merging one occurrence can invalidate its immediate neighbour.
        var path = WriteTempFile(corpus);
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { path }, targetVocabSize: 300, scratchDirectory: ScratchDirectory);

            var encoded = tokeniser.Encode(corpus);
            var decoded = tokeniser.Decode(encoded);

            Assert.Equal(corpus, decoded);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Train_MultipleFilesDoNotMergeAcrossFileBoundaries()
    {
        var pathA = WriteTempFile("xyxyxyxyxyxyxyxyxyxy");
        var pathB = WriteTempFile("zzzzzzzzzzzzzzzzzzzz");
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { pathA, pathB }, targetVocabSize: 260, scratchDirectory: ScratchDirectory);

            Assert.Equal("xyxyxyxyxyxyxyxyxyxy", tokeniser.Decode(tokeniser.Encode("xyxyxyxyxyxyxyxyxyxy")));
            Assert.Equal("zzzzzzzzzzzzzzzzzzzz", tokeniser.Decode(tokeniser.Encode("zzzzzzzzzzzzzzzzzzzz")));
        }
        finally
        {
            File.Delete(pathA);
            File.Delete(pathB);
        }
    }

    [Fact]
    public void SaveThenLoad_ProducesEquivalentTokeniser()
    {
        var corpus = "the quick brown fox jumps over the lazy dog. the dog barks. the fox runs.";
        var trainPath = WriteTempFile(corpus);
        var vocabPath = Path.GetTempFileName();
        try
        {
            var tokeniser = new BpeTokeniser();
            tokeniser.Train(new[] { trainPath }, targetVocabSize: 280, scratchDirectory: ScratchDirectory);
            tokeniser.Save(vocabPath);

            var loaded = BpeTokeniser.Load(vocabPath);

            Assert.Equal(tokeniser.VocabSize, loaded.VocabSize);

            var encodedOriginal = tokeniser.Encode(corpus);
            var encodedLoaded = loaded.Encode(corpus);
            Assert.Equal(encodedOriginal, encodedLoaded);
            Assert.Equal(corpus, loaded.Decode(encodedLoaded));
        }
        finally
        {
            File.Delete(trainPath);
            File.Delete(vocabPath);
        }
    }

    private static string WriteTempFile(string contents)
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, contents);
        return path;
    }
}
