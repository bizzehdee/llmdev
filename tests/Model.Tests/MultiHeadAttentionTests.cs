using Xunit;
using static Model.Tests.GradientCheck;

namespace Model.Tests;

public class MultiHeadAttentionTests
{
    [Fact]
    public void Constructor_ExposesConfiguredDimensions()
    {
        var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, random: new Random(1));

        Assert.Equal(8, mha.EmbeddingDim);
        Assert.Equal(2, mha.NumHeads);
        Assert.Equal(4, mha.HeadDim);
    }

    [Fact]
    public void Constructor_WithoutExplicitRandom_StillInitialises()
    {
        var mha = new MultiHeadAttention(embeddingDim: 4, numHeads: 2);

        Assert.Contains(mha.QueryWeight.Value.ToArray(), v => v != 0f);
    }

    [Fact]
    public void Constructor_EmbeddingDimNotDivisibleByNumHeadsThrows()
    {
        Assert.Throws<ArgumentException>(() => new MultiHeadAttention(embeddingDim: 10, numHeads: 3, random: new Random(1)));
    }

    [Fact]
    public void Forward_OutputShapeMatchesInputShape()
    {
        var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, random: new Random(1));
        var input = RandomVariable([5, 8]);

        var output = mha.Forward(input);

        Assert.Equal(new[] { 5, 8 }, output.Value.Shape);
    }

    [Fact]
    public void Forward_Causal_FuturePositionDoesNotAffectEarlierOutput()
    {
        var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, causal: true, random: new Random(1));
        var input = RandomVariable([5, 8]);
        var inputChanged = CloneWithChangedRow(input, rowToChange: 4);

        var outputBase = mha.Forward(input);
        var outputChanged = mha.Forward(inputChanged);

        for (int i = 0; i < 4; i++)
        {
            for (int d = 0; d < 8; d++)
            {
                Assert.Equal(outputBase.Value[i, d], outputChanged.Value[i, d], precision: 4);
            }
        }
    }

    [Fact]
    public void Forward_NonCausal_FuturePositionCanAffectEarlierOutput()
    {
        var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, causal: false, random: new Random(1));
        var input = RandomVariable([5, 8]);
        var inputChanged = CloneWithChangedRow(input, rowToChange: 4);

        var outputBase = mha.Forward(input);
        var outputChanged = mha.Forward(inputChanged);

        bool anyEarlierPositionChanged = false;
        for (int i = 0; i < 4; i++)
        {
            for (int d = 0; d < 8; d++)
            {
                if (MathF.Abs(outputBase.Value[i, d] - outputChanged.Value[i, d]) > 1e-4f)
                {
                    anyEarlierPositionChanged = true;
                }
            }
        }
        Assert.True(anyEarlierPositionChanged, "Expected non-causal attention to let an earlier position's output depend on a later position's input.");
    }

    [Fact]
    public void Forward_GradientFlowsIntoInput()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        AgainstParameter(input => mha.Forward(input), RandomVariable([3, 6]));
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_QueryWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        AgainstParameter(() => mha.Forward(input), mha.QueryWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_KeyWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        AgainstParameter(() => mha.Forward(input), mha.KeyWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_ValueWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        AgainstParameter(() => mha.Forward(input), mha.ValueWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_OutputWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        AgainstParameter(() => mha.Forward(input), mha.OutputWeight);
    }
}
