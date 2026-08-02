using Tensor;
using Xunit;

namespace Tensor.Tests;

public class VariableTests
{
    [Fact]
    public void Backward_ScalarChain_KnownDerivative()
    {
        // f(x) = x^2 + 3x, f'(x) = 2x + 3. At x=4: f'(4) = 11.
        var x = new Variable(Tensor.FromValues([4], [1]));
        var three = new Variable(Tensor.FromValues([3], [1]));

        var y = x.Multiply(x).Add(x.Multiply(three));
        y.Backward();

        Assert.Equal(11f, x.Gradient.ToArray()[0], precision: 4);
    }

    [Fact]
    public void Backward_VariableUsedTwice_AccumulatesGradientInsteadOfOverwriting()
    {
        // y = x + x, dy/dx = 2 (not 1 - this is the case that breaks a
        // backward pass that overwrites instead of accumulates gradients).
        var x = new Variable(Tensor.FromValues([5], [1]));

        var y = x.Add(x);
        y.Backward();

        Assert.Equal(2f, x.Gradient.ToArray()[0], precision: 4);
    }

    [Fact]
    public void Backward_DiamondDependency_AccumulatesFromBothPaths()
    {
        // b = a*2, c = a*3, d = b+c => d = 5a, dd/da = 5.
        var a = new Variable(Tensor.FromValues([2], [1]));
        var two = new Variable(Tensor.FromValues([2], [1]));
        var three = new Variable(Tensor.FromValues([3], [1]));

        var b = a.Multiply(two);
        var c = a.Multiply(three);
        var d = b.Add(c);
        d.Backward();

        Assert.Equal(5f, a.Gradient.ToArray()[0], precision: 4);
    }

    [Fact]
    public void ZeroGrad_ResetsGradientToZero()
    {
        var x = new Variable(Tensor.FromValues([2], [1]));
        var y = x.Multiply(x);
        y.Backward();
        Assert.NotEqual(0f, x.Gradient.ToArray()[0]);

        x.ZeroGrad();

        Assert.Equal(0f, x.Gradient.ToArray()[0]);
    }

    [Fact]
    public void Add_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].Add(vars[1]),
            RandomVariable([2, 3]), RandomVariable([2, 3]));
    }

    [Fact]
    public void Add_BroadcastGradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].Add(vars[1]),
            RandomVariable([2, 3]), RandomVariable([3]));
    }

    [Fact]
    public void Subtract_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].Subtract(vars[1]),
            RandomVariable([2, 3]), RandomVariable([2, 3]));
    }

    [Fact]
    public void Multiply_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].Multiply(vars[1]),
            RandomVariable([2, 3]), RandomVariable([2, 3]));
    }

    [Fact]
    public void Multiply_BroadcastGradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].Multiply(vars[1]),
            RandomVariable([2, 3]), RandomVariable([2, 1]));
    }

    [Fact]
    public void Divide_GradientMatchesFiniteDifference()
    {
        // Keep the denominator well away from zero.
        CheckGradient(
            vars => vars[0].Divide(vars[1]),
            RandomVariable([2, 3]), RandomVariable([2, 3], min: 2f, max: 5f));
    }

    [Fact]
    public void Negate_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Negate(), RandomVariable([2, 3]));
    }

    [Fact]
    public void MatMul_TwoDimensional_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].MatMul(vars[1]),
            RandomVariable([2, 3]), RandomVariable([3, 4]));
    }

    [Fact]
    public void MatMul_Batched_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].MatMul(vars[1]),
            RandomVariable([2, 3, 4]), RandomVariable([2, 4, 5]));
    }

    [Fact]
    public void MatMul_BroadcastBatch_GradientMatchesFiniteDifference()
    {
        CheckGradient(
            vars => vars[0].MatMul(vars[1]),
            RandomVariable([2, 3, 4]), RandomVariable([4, 5]));
    }

    [Fact]
    public void Transpose_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Transpose(0, 1), RandomVariable([2, 3]));
    }

    [Fact]
    public void Reshape_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Reshape([3, 2]), RandomVariable([2, 3]));
    }

    [Fact]
    public void Reshape_PreservesValuesInRowMajorOrder()
    {
        var x = new Variable(Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]));

        var y = x.Reshape([3, 2]);

        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, y.Value.ToArray());
    }

    [Fact]
    public void Sum_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Sum(axis: 1), RandomVariable([2, 3]));
    }

    [Fact]
    public void Sum_KeepDims_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Sum(axis: 0, keepDims: true), RandomVariable([2, 3]));
    }

    [Fact]
    public void Mean_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Mean(axis: 1), RandomVariable([2, 3]));
    }

    [Fact]
    public void Exp_GradientMatchesFiniteDifference()
    {
        // Keep inputs small so exp() doesn't blow up finite-difference precision.
        CheckGradient(vars => vars[0].Exp(), RandomVariable([2, 3], min: -1f, max: 1f));
    }

    [Fact]
    public void Sqrt_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Sqrt(), RandomVariable([2, 3], min: 0.5f, max: 4f));
    }

    [Fact]
    public void Sqrt_MatchesMathSqrt()
    {
        var x = new Variable(Tensor.FromValues([4, 9, 16], [3]));

        var y = x.Sqrt();

        Assert.Equal(new float[] { 2, 3, 4 }, y.Value.ToArray());
    }

    [Fact]
    public void Log_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Log(), RandomVariable([2, 3], min: 0.5f, max: 3f));
    }

    [Fact]
    public void Relu_GradientMatchesFiniteDifference()
    {
        // Avoid values near 0, where Relu is non-differentiable.
        CheckGradient(vars => vars[0].Relu(), RandomVariable([2, 3], min: 0.5f, max: 3f));
        CheckGradient(vars => vars[0].Relu(), RandomVariable([2, 3], min: -3f, max: -0.5f));
    }

    [Fact]
    public void Softmax_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].Softmax(axis: 1), RandomVariable([2, 3], min: -1f, max: 1f));
    }

    [Fact]
    public void Softmax_SumsToOneAlongAxis()
    {
        var x = new Variable(Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]));

        var y = x.Softmax(axis: 1);

        var rowSums = y.Value.Sum(axis: 1).ToArray();
        Assert.All(rowSums, s => Assert.Equal(1f, s, precision: 5));
    }

    [Fact]
    public void ChainedOps_GradientMatchesFiniteDifference()
    {
        // A small stand-in for a linear layer + activation + reduction,
        // i.e. the shape of computation a transformer block will chain
        // together: matmul -> add bias -> relu -> mean.
        CheckGradient(
            vars => vars[0].MatMul(vars[1]).Add(vars[2]).Relu().Mean(axis: 1),
            RandomVariable([2, 3], min: 0.5f, max: 2f),
            RandomVariable([3, 4], min: 0.5f, max: 2f),
            RandomVariable([4], min: 0.5f, max: 2f));
    }

    [Fact]
    public void GatherRows_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].GatherRows([2, 0]), RandomVariable([3, 2]));
    }

    [Fact]
    public void GatherRows_RepeatedIndex_AccumulatesGradientForThatRow()
    {
        // Row 0 is looked up twice, so its gradient should be the sum of
        // both occurrences' upstream gradients, not just the last one.
        var weights = new Variable(Tensor.FromValues([1, 2, 3, 4], [2, 2]));

        var looked = weights.GatherRows([0, 0]);
        looked.Backward();

        Assert.Equal(new float[] { 2, 2, 0, 0 }, weights.Gradient.ToArray());
    }

    [Fact]
    public void GatherColumns_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => vars[0].GatherColumns([1, 0]), RandomVariable([2, 3]));
    }

    private static readonly Random Rng = new(1234);

    private static Variable RandomVariable(int[] shape, float min = -3f, float max = 3f)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = min + (float)Rng.NextDouble() * (max - min);
        }
        return new Variable(Tensor.FromValues(values, shape));
    }

    /// <summary>
    /// Verifies analytic gradients (from Backward()) against a central-difference
    /// numerical approximation of d(sum(forward(inputs)))/d(input) for every
    /// input element - the standard way to test autodiff correctness without
    /// hand-deriving an expected gradient for every op.
    /// </summary>
    private static void CheckGradient(Func<Variable[], Variable> forward, params Variable[] inputs)
    {
        const float epsilon = 1e-3f;
        const float tolerance = 5e-2f;

        foreach (var v in inputs)
        {
            v.ZeroGrad();
        }
        var output = forward(inputs);
        output.Backward();

        for (int p = 0; p < inputs.Length; p++)
        {
            var original = inputs[p].Value.ToArray();
            var analytic = inputs[p].Gradient.ToArray();

            for (int i = 0; i < original.Length; i++)
            {
                float numeric = NumericPartialDerivative(forward, inputs, p, i, original, epsilon);

                Assert.True(
                    MathF.Abs(numeric - analytic[i]) < tolerance,
                    $"Gradient mismatch for input {p}, element {i}: analytic={analytic[i]}, numeric={numeric}");
            }
        }
    }

    private static float NumericPartialDerivative(Func<Variable[], Variable> forward, Variable[] inputs, int paramIndex, int elementIndex, float[] original, float epsilon)
    {
        float sumPlus = EvaluateSumWithPerturbation(forward, inputs, paramIndex, elementIndex, original, epsilon);
        float sumMinus = EvaluateSumWithPerturbation(forward, inputs, paramIndex, elementIndex, original, -epsilon);
        return (sumPlus - sumMinus) / (2 * epsilon);
    }

    private static float EvaluateSumWithPerturbation(Func<Variable[], Variable> forward, Variable[] inputs, int paramIndex, int elementIndex, float[] original, float delta)
    {
        var perturbedValues = (float[])original.Clone();
        perturbedValues[elementIndex] += delta;

        var perturbedInputs = (Variable[])inputs.Clone();
        perturbedInputs[paramIndex] = new Variable(Tensor.FromValues(perturbedValues, inputs[paramIndex].Value.Shape));

        return forward(perturbedInputs).Value.ToArray().Sum();
    }
}
