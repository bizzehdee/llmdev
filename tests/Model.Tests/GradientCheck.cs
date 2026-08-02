using Tensor;
using Xunit;
using TensorValue = Tensor.Tensor;

namespace Model.Tests;

/// <summary>
/// Shared finite-difference gradient checking for Model.Tests, factored
/// out of MultiHeadAttentionTests since LayerNorm/FeedForward/
/// TransformerBlock all need the same check against parameters that live
/// inside a persistent module (as opposed to VariableTests.CheckGradient in
/// Tensor.Tests, which only ever checks freshly constructed leaf variables).
/// </summary>
internal static class GradientCheck
{
    private static readonly Random Rng = new(99);

    public static Variable RandomVariable(int[] shape, float min = -1f, float max = 1f)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = min + (float)Rng.NextDouble() * (max - min);
        }
        return new Variable(TensorValue.FromValues(values, shape));
    }

    public static Variable CloneWithChangedRow(Variable source, int rowToChange, float delta = 10f)
    {
        var values = source.Value.ToArray();
        int cols = source.Value.Shape[1];
        for (int d = 0; d < cols; d++)
        {
            values[rowToChange * cols + d] += delta;
        }
        return new Variable(TensorValue.FromValues(values, source.Value.Shape));
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
    /// perturbs the target's existing Tensor values in place (via Tensor's
    /// indexer setter) rather than substituting a freshly constructed
    /// Variable, since the target here can't simply be swapped out of the
    /// module that owns it.
    /// </summary>
    public static void AgainstParameter(Func<Variable, Variable> forward, Variable target)
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
    public static void AgainstParameter(Func<Variable> forward, Variable target) =>
        AgainstParameter(_ => forward(), target);
}
