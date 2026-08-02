using Xunit;
using static Model.Tests.GradientCheck;

namespace Model.Tests;

public class FeedForwardTests
{
    [Fact]
    public void Constructor_ExposesConfiguredDimensions()
    {
        var ff = new FeedForward(embeddingDim: 6, hiddenDim: 24, new Random(1));

        Assert.Equal(6, ff.EmbeddingDim);
        Assert.Equal(24, ff.HiddenDim);
        Assert.Equal(new[] { 6, 24 }, ff.InputWeight.Value.Shape);
        Assert.Equal(new[] { 24 }, ff.InputBias.Value.Shape);
        Assert.Equal(new[] { 24, 6 }, ff.OutputWeight.Value.Shape);
        Assert.Equal(new[] { 6 }, ff.OutputBias.Value.Shape);
    }

    [Fact]
    public void Constructor_BiasesInitialiseToZero()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 16, new Random(1));

        Assert.All(ff.InputBias.Value.ToArray(), v => Assert.Equal(0f, v));
        Assert.All(ff.OutputBias.Value.ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Forward_OutputShapeMatchesInputShape()
    {
        var ff = new FeedForward(embeddingDim: 6, hiddenDim: 24, new Random(1));
        var input = RandomVariable([5, 6]);

        var output = ff.Forward(input);

        Assert.Equal(new[] { 5, 6 }, output.Value.Shape);
    }

    [Fact]
    public void Forward_IsAppliedIndependentlyPerPosition()
    {
        // No cross-position mixing: changing one row of the input must not
        // change any other row of the output.
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 16, new Random(1));
        var input = RandomVariable([3, 4]);
        var inputChanged = CloneWithChangedRow(input, rowToChange: 1);

        var outputBase = ff.Forward(input);
        var outputChanged = ff.Forward(inputChanged);

        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(outputBase.Value[0, d], outputChanged.Value[0, d], precision: 4);
            Assert.Equal(outputBase.Value[2, d], outputChanged.Value[2, d], precision: 4);
        }
    }

    [Fact]
    public void Forward_GradientFlowsIntoInput()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 16, new Random(1));
        AgainstParameter(x => ff.Forward(x), RandomVariable([3, 4]));
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_InputWeight()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 8, new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => ff.Forward(input), ff.InputWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_InputBias()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 8, new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => ff.Forward(input), ff.InputBias);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_OutputWeight()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 8, new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => ff.Forward(input), ff.OutputWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_OutputBias()
    {
        var ff = new FeedForward(embeddingDim: 4, hiddenDim: 8, new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => ff.Forward(input), ff.OutputBias);
    }
}
