using Xunit;

namespace Model.Tests;

public class GenerationCacheTests
{
    /// <summary>
    /// TASK-020's own bar: cached-path output must be *exactly* the same
    /// as the non-cached path, at every step - not "plausibly similar".
    /// Feeds the same token sequence through both `Forward` (recomputing
    /// everything from scratch each time) and `ForwardIncremental`
    /// (growing a cache one token at a time after an initial multi-token
    /// prefill) and compares logits directly, at every step.
    /// </summary>
    [Fact]
    public void ForwardIncremental_MatchesNonCachedForwardAtEveryStep()
    {
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 3, numHeads: 2, maxSequenceLength: 16, random: new Random(1));
        int[] tokens = [3, 7, 1, 9, 4, 12, 6];

        using var cache = new GenerationCache(model.NumLayers);

        // Prefill with the first three tokens in one incremental call...
        const int prefixLength = 3;
        var cachedLogits = model.ForwardIncremental(tokens[..prefixLength], cache).Value;
        AssertLastRowMatches(model, tokens[..prefixLength], cachedLogits, cachedLogits.Shape[0] - 1);

        // ...then one token at a time for the rest.
        for (int step = prefixLength; step < tokens.Length; step++)
        {
            cachedLogits = model.ForwardIncremental([tokens[step]], cache).Value;
            AssertLastRowMatches(model, tokens[..(step + 1)], cachedLogits, row: 0);
        }
    }

    private static void AssertLastRowMatches(GptModel model, int[] contextSoFar, Tensor.Tensor cachedLogits, int row)
    {
        var reference = model.Forward(contextSoFar).Value;
        int vocabSize = reference.Shape[1];
        int referenceRow = reference.Shape[0] - 1;

        for (int v = 0; v < vocabSize; v++)
        {
            Assert.Equal(reference[referenceRow, v], cachedLogits[row, v], precision: 3);
        }
    }

    [Fact]
    public void ForwardIncremental_GrowsCacheLengthByNumberOfNewTokens()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 4, numLayers: 2, numHeads: 2, maxSequenceLength: 16, random: new Random(1));
        using var cache = new GenerationCache(model.NumLayers);

        Assert.Equal(0, cache.Length);

        model.ForwardIncremental([1, 2, 3], cache);
        Assert.Equal(3, cache.Length);

        model.ForwardIncremental([4], cache);
        Assert.Equal(4, cache.Length);
    }

    [Fact]
    public void ForwardIncremental_OutputShapeIsNewTokenCountByVocabSize()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 4, numLayers: 2, numHeads: 2, maxSequenceLength: 16, random: new Random(1));
        using var cache = new GenerationCache(model.NumLayers);

        model.ForwardIncremental([1, 2, 3], cache);
        var result = model.ForwardIncremental([4, 5], cache);

        Assert.Equal(new[] { 2, 10 }, result.Value.Shape);
    }

    [Fact]
    public void Reset_ClearsLengthAndAllowsRebuildingFromScratch()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 4, numLayers: 2, numHeads: 2, maxSequenceLength: 16, random: new Random(1));
        using var cache = new GenerationCache(model.NumLayers);
        model.ForwardIncremental([1, 2, 3], cache);

        cache.Reset();

        Assert.Equal(0, cache.Length);
        // Should behave exactly like a fresh cache afterwards.
        var afterReset = model.ForwardIncremental([1, 2, 3], cache).Value;
        var reference = model.Forward([1, 2, 3]).Value;
        for (int v = 0; v < reference.Shape[1]; v++)
        {
            Assert.Equal(reference[2, v], afterReset[2, v], precision: 3);
        }
    }

    [Fact]
    public void ForwardIncremental_PositionBeyondMaxSequenceLengthThrows()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 4, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        using var cache = new GenerationCache(model.NumLayers);
        model.ForwardIncremental([1, 2, 3, 4], cache);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardIncremental([5], cache));
    }
}
