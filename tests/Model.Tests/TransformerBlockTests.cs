using Xunit;
using static Model.Tests.GradientCheck;

namespace Model.Tests;

public class TransformerBlockTests
{
    [Fact]
    public void Constructor_DefaultsFeedForwardHiddenDimToFourTimesEmbeddingDim()
    {
        var block = new TransformerBlock(embeddingDim: 8, numHeads: 2, random: new Random(1));

        Assert.Equal(32, block.FeedForward.HiddenDim);
    }

    [Fact]
    public void Constructor_WithoutExplicitRandom_StillInitialises()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2);

        Assert.Contains(block.Attention.QueryWeight.Value.ToArray(), v => v != 0f);
    }

    [Fact]
    public void Constructor_FeedForwardHiddenDimIsOverridable()
    {
        var block = new TransformerBlock(embeddingDim: 8, numHeads: 2, feedForwardHiddenDim: 10, random: new Random(1));

        Assert.Equal(10, block.FeedForward.HiddenDim);
    }

    [Fact]
    public void Forward_OutputShapeMatchesInputShape()
    {
        var block = new TransformerBlock(embeddingDim: 8, numHeads: 2, random: new Random(1));
        var input = RandomVariable([6, 8]);

        var output = block.Forward(input);

        Assert.Equal(new[] { 6, 8 }, output.Value.Shape);
    }

    [Fact]
    public void Forward_Causal_FuturePositionDoesNotAffectEarlierOutput()
    {
        // End-to-end check that causal masking survives being wrapped in a
        // full block (residual connections and layernorm don't leak future
        // information back into earlier positions).
        var block = new TransformerBlock(embeddingDim: 8, numHeads: 2, causal: true, random: new Random(1));
        var input = RandomVariable([5, 8]);
        var inputChanged = CloneWithChangedRow(input, rowToChange: 4);

        var outputBase = block.Forward(input);
        var outputChanged = block.Forward(inputChanged);

        for (int i = 0; i < 4; i++)
        {
            for (int d = 0; d < 8; d++)
            {
                Assert.Equal(outputBase.Value[i, d], outputChanged.Value[i, d], precision: 4);
            }
        }
    }

    [Fact]
    public void Forward_ResidualConnection_OutputDiffersFromPureSubLayerOutput()
    {
        // Sanity check that the residual add is actually wired in: with
        // all-zero attention/feed-forward weights the block's output would
        // just be layernorm noise if the residual were missing, but with it
        // wired in, the output should track the (layernormed magnitude of
        // the) input rather than collapsing towards the sub-layer biases.
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 4], min: 5f, max: 6f); // large, distinctly-signed input

        var output = block.Forward(input);

        // If the residual were dropped, output would be independent of how
        // large/positive the input is; instead it should differ noticeably
        // between two very different input scales.
        var inputOther = RandomVariable([3, 4], min: -6f, max: -5f);
        var outputOther = block.Forward(inputOther);

        bool anyDifference = false;
        for (int i = 0; i < 3; i++)
        {
            for (int d = 0; d < 4; d++)
            {
                if (MathF.Abs(output.Value[i, d] - outputOther.Value[i, d]) > 1e-3f)
                {
                    anyDifference = true;
                }
            }
        }
        Assert.True(anyDifference, "Expected the block's output to depend on the input, not just learned biases.");
    }

    [Fact]
    public void Forward_GradientFlowsIntoInput()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, random: new Random(1));
        AgainstParameter(x => block.Forward(x), RandomVariable([3, 4]));
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_AttentionQueryWeight()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => block.Forward(input), block.Attention.QueryWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_FeedForwardInputWeight()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, feedForwardHiddenDim: 8, random: new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => block.Forward(input), block.FeedForward.InputWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_PreAttentionNormGamma()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => block.Forward(input), block.PreAttentionNorm.Gamma);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_PreFeedForwardNormBeta()
    {
        var block = new TransformerBlock(embeddingDim: 4, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => block.Forward(input), block.PreFeedForwardNorm.Beta);
    }
}
