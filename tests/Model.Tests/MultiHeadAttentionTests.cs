using Tensor;
using Xunit;
using TensorValue = Tensor.Tensor;

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
        CheckGradientAgainstParameter(
            input => mha.Forward(input),
            RandomVariable([3, 6]));
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_QueryWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        CheckGradientAgainstParameter(() => mha.Forward(input), mha.QueryWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_KeyWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        CheckGradientAgainstParameter(() => mha.Forward(input), mha.KeyWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_ValueWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        CheckGradientAgainstParameter(() => mha.Forward(input), mha.ValueWeight);
    }

    [Fact]
    public void Forward_GradientMatchesFiniteDifference_OutputWeight()
    {
        var mha = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, random: new Random(1));
        var input = RandomVariable([3, 6]);
        CheckGradientAgainstParameter(() => mha.Forward(input), mha.OutputWeight);
    }

    private static Variable CloneWithChangedRow(Variable source, int rowToChange)
    {
        var values = source.Value.ToArray();
        int cols = source.Value.Shape[1];
        for (int d = 0; d < cols; d++)
        {
            values[rowToChange * cols + d] += 10f;
        }
        return new Variable(TensorValue.FromValues(values, source.Value.Shape));
    }

    private static readonly Random Rng = new(99);

    private static Variable RandomVariable(int[] shape, float min = -1f, float max = 1f)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = min + (float)Rng.NextDouble() * (max - min);
        }
        return new Variable(TensorValue.FromValues(values, shape));
    }

    private static int[] FlatToCoords(int flatIndex, int[] shape)
    {
        var coords = new int[shape.Length];
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            coords[d] = flatIndex % shape[d];
            flatIndex /= shape[d];
        }
        return coords;
    }

    /// <summary>
    /// Finite-difference gradient check against a parameter Variable that
    /// already lives inside a persistent module (e.g. mha.QueryWeight) -
    /// unlike VariableTests.CheckGradient, this perturbs the target's
    /// existing Tensor values in place (via Tensor's indexer setter) rather
    /// than substituting a freshly constructed Variable, since the target
    /// here can't simply be swapped out of the module that owns it.
    /// </summary>
    private static void CheckGradientAgainstParameter(Func<Variable, Variable> forward, Variable target)
    {
        const float epsilon = 1e-3f;
        const float tolerance = 5e-2f;

        target.ZeroGrad();
        var output = forward(target);
        output.Backward();
        var analytic = target.Gradient.ToArray();

        var shape = target.Value.Shape;
        for (int flat = 0; flat < target.Value.Length; flat++)
        {
            var coords = FlatToCoords(flat, shape);
            float original = target.Value[coords];

            target.Value[coords] = original + epsilon;
            float sumPlus = forward(target).Value.ToArray().Sum();

            target.Value[coords] = original - epsilon;
            float sumMinus = forward(target).Value.ToArray().Sum();

            target.Value[coords] = original;

            float numeric = (sumPlus - sumMinus) / (2 * epsilon);
            Assert.True(
                MathF.Abs(numeric - analytic[flat]) < tolerance,
                $"Gradient mismatch at flat index {flat}: analytic={analytic[flat]}, numeric={numeric}");
        }
    }

    /// <summary>Overload for checking gradient w.r.t. a module's owned weight rather than its input.</summary>
    private static void CheckGradientAgainstParameter(Func<Variable> forward, Variable target)
    {
        CheckGradientAgainstParameter(_ => forward(), target);
    }
}
