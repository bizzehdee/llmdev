using Model;
using Tensor;
using Xunit;

namespace Model.Tests;

public class TokenEmbeddingTests
{
    [Fact]
    public void Constructor_ExposesVocabSizeAndEmbeddingDim()
    {
        var embedding = new TokenEmbedding(vocabSize: 10, embeddingDim: 4, new Random(1));

        Assert.Equal(10, embedding.VocabSize);
        Assert.Equal(4, embedding.EmbeddingDim);
        Assert.Equal(new[] { 10, 4 }, embedding.Weight.Value.Shape);
    }

    [Fact]
    public void Constructor_WithoutExplicitRandom_StillInitialises()
    {
        var embedding = new TokenEmbedding(vocabSize: 10, embeddingDim: 4);

        Assert.Contains(embedding.Weight.Value.ToArray(), v => v != 0f);
    }

    [Fact]
    public void Constructor_InitialisesWithSmallNonZeroValues()
    {
        var embedding = new TokenEmbedding(vocabSize: 50, embeddingDim: 16, new Random(1));

        var values = embedding.Weight.Value.ToArray();

        Assert.Contains(values, v => v != 0f);
        Assert.All(values, v => Assert.True(MathF.Abs(v) < 1f, $"Expected small init values, got {v}"));
    }

    [Fact]
    public void Constructor_SameSeedProducesSameWeights()
    {
        var a = new TokenEmbedding(vocabSize: 20, embeddingDim: 8, new Random(42));
        var b = new TokenEmbedding(vocabSize: 20, embeddingDim: 8, new Random(42));

        Assert.Equal(a.Weight.Value.ToArray(), b.Weight.Value.ToArray());
    }

    [Fact]
    public void Forward_ReturnsOneRowPerTokenInOrder()
    {
        var embedding = new TokenEmbedding(vocabSize: 5, embeddingDim: 3, new Random(1));

        var result = embedding.Forward([2, 0, 4]);

        Assert.Equal(new[] { 3, 3 }, result.Value.Shape);
        for (int i = 0; i < 3; i++)
        {
            int tokenId = new[] { 2, 0, 4 }[i];
            for (int d = 0; d < 3; d++)
            {
                Assert.Equal(embedding.Weight.Value[tokenId, d], result.Value[i, d]);
            }
        }
    }

    [Fact]
    public void Forward_OutOfRangeTokenIdThrows()
    {
        var embedding = new TokenEmbedding(vocabSize: 5, embeddingDim: 3, new Random(1));

        Assert.Throws<IndexOutOfRangeException>(() => embedding.Forward([5]));
    }

    [Fact]
    public void Backward_OnlyUpdatesLookedUpRowsAndAccumulatesRepeats()
    {
        var embedding = new TokenEmbedding(vocabSize: 4, embeddingDim: 2, new Random(1));

        // Token 1 appears twice, token 3 once, tokens 0 and 2 are never looked up.
        var result = embedding.Forward([1, 3, 1]);
        result.Sum(axis: 0).Sum(axis: 0).Backward();

        var grad = embedding.Weight.Gradient.ToArray();

        // Row 0 (token 0, unused): zero gradient.
        Assert.Equal(0f, grad[0]);
        Assert.Equal(0f, grad[1]);
        // Row 1 (token 1, used twice): gradient of 2 per dimension (d(sum)/dx = 1 per occurrence).
        Assert.Equal(2f, grad[2]);
        Assert.Equal(2f, grad[3]);
        // Row 2 (token 2, unused): zero gradient.
        Assert.Equal(0f, grad[4]);
        Assert.Equal(0f, grad[5]);
        // Row 3 (token 3, used once): gradient of 1 per dimension.
        Assert.Equal(1f, grad[6]);
        Assert.Equal(1f, grad[7]);
    }
}
