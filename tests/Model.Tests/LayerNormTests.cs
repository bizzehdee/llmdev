using Tensor;
using Xunit;
using static Model.Tests.GradientCheck;
using TensorValue = Tensor.Tensor;

namespace Model.Tests;

public class LayerNormTests
{
    [Fact]
    public void Constructor_InitialisesGammaToOneAndBetaToZero()
    {
        var norm = new LayerNorm(embeddingDim: 4);

        Assert.Equal(new float[] { 1, 1, 1, 1 }, norm.Gamma.Value.ToArray());
        Assert.Equal(new float[] { 0, 0, 0, 0 }, norm.Beta.Value.ToArray());
    }

    [Fact]
    public void Forward_OutputShapeMatchesInputShape()
    {
        var norm = new LayerNorm(embeddingDim: 6);
        var input = RandomVariable([5, 6]);

        var output = norm.Forward(input);

        Assert.Equal(new[] { 5, 6 }, output.Value.Shape);
    }

    [Fact]
    public void Forward_WithDefaultGammaBeta_NormalisesEachRowToZeroMeanUnitVariance()
    {
        var norm = new LayerNorm(embeddingDim: 6);
        var input = RandomVariable([4, 6], min: -5f, max: 5f);

        var output = norm.Forward(input);

        for (int row = 0; row < 4; row++)
        {
            float sum = 0f, sumSquares = 0f;
            for (int d = 0; d < 6; d++)
            {
                float v = output.Value[row, d];
                sum += v;
                sumSquares += v * v;
            }
            float mean = sum / 6;
            float variance = sumSquares / 6 - mean * mean;

            Assert.True(MathF.Abs(mean) < 1e-3f, $"Row {row} mean was {mean}, expected ~0.");
            Assert.True(MathF.Abs(variance - 1f) < 1e-2f, $"Row {row} variance was {variance}, expected ~1.");
        }
    }

    [Fact]
    public void Forward_IsAppliedIndependentlyPerRow()
    {
        // Changing one row's input must not change any other row's output -
        // normalisation is per position, not across the whole sequence.
        var norm = new LayerNorm(embeddingDim: 4);
        var input = RandomVariable([3, 4]);
        var inputChanged = CloneWithChangedRow(input, rowToChange: 1);

        var outputBase = norm.Forward(input);
        var outputChanged = norm.Forward(inputChanged);

        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(outputBase.Value[0, d], outputChanged.Value[0, d], precision: 4);
            Assert.Equal(outputBase.Value[2, d], outputChanged.Value[2, d], precision: 4);
        }
    }

    [Fact]
    public void Forward_GradientFlowsIntoInput()
    {
        var norm = new LayerNorm(embeddingDim: 4);
        AgainstParameter(x => norm.Forward(x), RandomVariable([3, 4]));
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_Gamma()
    {
        var norm = new LayerNorm(embeddingDim: 4);
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => norm.Forward(input), norm.Gamma);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_Beta()
    {
        var norm = new LayerNorm(embeddingDim: 4);
        var input = RandomVariable([3, 4]);
        AgainstParameter(() => norm.Forward(input), norm.Beta);
    }

    [Fact]
    public void Forward_NonDefaultGammaBetaScalesAndShiftsNormalisedOutput()
    {
        var norm = new LayerNorm(embeddingDim: 3);
        norm.Gamma.Value[0] = 2f;
        norm.Gamma.Value[1] = 2f;
        norm.Gamma.Value[2] = 2f;
        norm.Beta.Value[0] = 5f;
        norm.Beta.Value[1] = 5f;
        norm.Beta.Value[2] = 5f;
        var input = new Variable(TensorValue.FromValues([1, 2, 3], [1, 3]));

        var output = norm.Forward(input);

        // With gamma=2, beta=5 applied uniformly: output = 2*normalised + 5,
        // so every output value must be >= 5 - 2*3 (loosely) and centred on 5.
        float mean = (output.Value[0, 0] + output.Value[0, 1] + output.Value[0, 2]) / 3f;
        Assert.Equal(5f, mean, precision: 3);
    }
}
