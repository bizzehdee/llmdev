using Model;
using Xunit;

namespace Model.Tests;

public class PositionalEmbeddingTests
{
    [Fact]
    public void Constructor_ExposesMaxSequenceLengthAndEmbeddingDim()
    {
        var embedding = new PositionalEmbedding(maxSequenceLength: 16, embeddingDim: 4, new Random(1));

        Assert.Equal(16, embedding.MaxSequenceLength);
        Assert.Equal(4, embedding.EmbeddingDim);
        Assert.Equal(new[] { 16, 4 }, embedding.Weight.Value.Shape);
    }

    [Fact]
    public void Constructor_InitialisesWithSmallNonZeroValues()
    {
        var embedding = new PositionalEmbedding(maxSequenceLength: 16, embeddingDim: 8, new Random(1));

        var values = embedding.Weight.Value.ToArray();

        Assert.Contains(values, v => v != 0f);
        Assert.All(values, v => Assert.True(MathF.Abs(v) < 1f, $"Expected small init values, got {v}"));
    }

    [Fact]
    public void Forward_ReturnsFirstNPositionsInOrder()
    {
        var embedding = new PositionalEmbedding(maxSequenceLength: 10, embeddingDim: 3, new Random(1));

        var result = embedding.Forward(sequenceLength: 4);

        Assert.Equal(new[] { 4, 3 }, result.Value.Shape);
        for (int pos = 0; pos < 4; pos++)
        {
            for (int d = 0; d < 3; d++)
            {
                Assert.Equal(embedding.Weight.Value[pos, d], result.Value[pos, d]);
            }
        }
    }

    [Fact]
    public void Forward_SequenceLengthBeyondMaxThrows()
    {
        var embedding = new PositionalEmbedding(maxSequenceLength: 4, embeddingDim: 3, new Random(1));

        Assert.Throws<ArgumentOutOfRangeException>(() => embedding.Forward(sequenceLength: 5));
    }

    [Fact]
    public void Forward_ZeroLengthReturnsEmptySequence()
    {
        var embedding = new PositionalEmbedding(maxSequenceLength: 4, embeddingDim: 3, new Random(1));

        var result = embedding.Forward(sequenceLength: 0);

        Assert.Equal(new[] { 0, 3 }, result.Value.Shape);
    }

    [Fact]
    public void CombinesWithTokenEmbeddingByElementwiseAddition()
    {
        var tokenEmbedding = new TokenEmbedding(vocabSize: 10, embeddingDim: 4, new Random(1));
        var positionalEmbedding = new PositionalEmbedding(maxSequenceLength: 8, embeddingDim: 4, new Random(2));
        int[] tokenIds = [3, 7, 1];

        var tokens = tokenEmbedding.Forward(tokenIds);
        var positions = positionalEmbedding.Forward(tokenIds.Length);
        var combined = tokens.Add(positions);

        Assert.Equal(new[] { 3, 4 }, combined.Value.Shape);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            for (int d = 0; d < 4; d++)
            {
                float expected = tokens.Value[i, d] + positions.Value[i, d];
                Assert.Equal(expected, combined.Value[i, d]);
            }
        }
    }

    [Fact]
    public void CombinedEmbedding_GradientFlowsToBothTables()
    {
        var tokenEmbedding = new TokenEmbedding(vocabSize: 5, embeddingDim: 2, new Random(1));
        var positionalEmbedding = new PositionalEmbedding(maxSequenceLength: 5, embeddingDim: 2, new Random(2));

        var combined = tokenEmbedding.Forward([2, 4]).Add(positionalEmbedding.Forward(2));
        combined.Sum(axis: 0).Sum(axis: 0).Backward();

        Assert.Contains(tokenEmbedding.Weight.Gradient.ToArray(), g => g != 0f);
        Assert.Contains(positionalEmbedding.Weight.Gradient.ToArray(), g => g != 0f);
    }
}
