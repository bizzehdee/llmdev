using Xunit;
using static Model.Tests.GradientCheck;

namespace Model.Tests;

public class GptModelTests
{
    [Fact]
    public void Constructor_ExposesConfiguredDimensions()
    {
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 3, numHeads: 2, maxSequenceLength: 16, random: new Random(1));

        Assert.Equal(20, model.VocabSize);
        Assert.Equal(8, model.EmbeddingDim);
        Assert.Equal(3, model.NumLayers);
        Assert.Equal(2, model.NumHeads);
        Assert.Equal(16, model.MaxSequenceLength);
        Assert.Equal(3, model.Blocks.Count);
    }

    [Fact]
    public void Forward_OutputShapeIsSequenceLengthByVocabSize()
    {
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 16, random: new Random(1));

        var logits = model.Forward([3, 7, 1, 9]);

        Assert.Equal(new[] { 4, 20 }, logits.Value.Shape);
    }

    [Fact]
    public void Forward_SequenceLongerThanMaxThrows()
    {
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 4, random: new Random(1));

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward([1, 2, 3, 4, 5]));
    }

    [Fact]
    public void Forward_CausalMaskingSurvivesTheFullModel()
    {
        // The end-to-end version of every earlier causal-masking check: a
        // later token must not change logits for any earlier position, even
        // after flowing through embeddings, every stacked block, and the
        // final norm + output projection.
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 3, numHeads: 2, maxSequenceLength: 16, random: new Random(1));
        int[] tokenIds = [3, 7, 1, 9, 5];
        int[] tokenIdsChanged = [3, 7, 1, 9, 12]; // only the last token differs

        var logitsBase = model.Forward(tokenIds);
        var logitsChanged = model.Forward(tokenIdsChanged);

        for (int i = 0; i < 4; i++)
        {
            for (int v = 0; v < 20; v++)
            {
                Assert.Equal(logitsBase.Value[i, v], logitsChanged.Value[i, v], precision: 3);
            }
        }
    }

    [Fact]
    public void Forward_DifferentTokenSequencesProduceDifferentLogits()
    {
        var model = new GptModel(vocabSize: 20, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 16, random: new Random(1));

        var logitsA = model.Forward([1, 2, 3]);
        var logitsB = model.Forward([4, 5, 6]);

        Assert.NotEqual(logitsA.Value.ToArray(), logitsB.Value.ToArray());
    }

    [Fact]
    public void Forward_WeightTying_OutputProjectionGradientReachesEmbeddingRowsNeverLookedUp()
    {
        // If the output projection genuinely reuses TokenEmbedding.Weight
        // (rather than an independent matrix), every vocabulary row
        // contributes to every logit via the transposed matmul - including
        // rows for tokens that never appeared in the input sequence. A
        // gradient reaching those untouched rows is proof the tying is real,
        // not just equal-by-coincidence initial values.
        var model = new GptModel(vocabSize: 10, embeddingDim: 4, numLayers: 1, numHeads: 2, maxSequenceLength: 8, random: new Random(1));
        int[] tokenIds = [1, 2]; // tokens 0, 3-9 never looked up

        var logits = model.Forward(tokenIds);
        logits.Sum(axis: 0).Sum(axis: 0).Backward();

        var grad = model.TokenEmbedding.Weight.Gradient;
        bool unlookedRowHasGradient = false;
        foreach (int unlookedToken in new[] { 0, 5, 9 })
        {
            for (int d = 0; d < 4; d++)
            {
                if (grad[unlookedToken, d] != 0f)
                {
                    unlookedRowHasGradient = true;
                }
            }
        }
        Assert.True(unlookedRowHasGradient, "Expected the output projection's gradient to reach embedding rows never looked up as input tokens, proving weight tying.");
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_TokenEmbeddingWeight()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 4, numLayers: 1, numHeads: 2, maxSequenceLength: 5, random: new Random(1));
        int[] tokenIds = [1, 3];
        AgainstParameter(() => model.Forward(tokenIds), model.TokenEmbedding.Weight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_PositionalEmbeddingWeight()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 4, numLayers: 1, numHeads: 2, maxSequenceLength: 5, random: new Random(1));
        int[] tokenIds = [1, 3];
        AgainstParameter(() => model.Forward(tokenIds), model.PositionalEmbedding.Weight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_FirstBlockAttentionWeight()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 4, numLayers: 2, numHeads: 2, maxSequenceLength: 5, random: new Random(1));
        int[] tokenIds = [1, 3];
        AgainstParameter(() => model.Forward(tokenIds), model.Blocks[0].Attention.QueryWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_FinalNormGamma()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 4, numLayers: 2, numHeads: 2, maxSequenceLength: 5, random: new Random(1));
        int[] tokenIds = [1, 3];
        AgainstParameter(() => model.Forward(tokenIds), model.FinalNorm.Gamma);
    }
}
