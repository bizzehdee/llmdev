namespace Tensor;

public sealed partial class Variable
{
    public Variable Sum(int axis, bool keepDims = false)
    {
        var value = Value.Sum(axis, keepDims);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(ExpandReducedAxis(result.Gradient, axis, keepDims));
        }, this);
    }

    public Variable Mean(int axis, bool keepDims = false)
    {
        var value = Value.Mean(axis, keepDims);
        float divisor = Value.Shape[axis];
        return FromOp(value, result => () =>
        {
            var expanded = ExpandReducedAxis(result.Gradient, axis, keepDims);
            var divisorTensor = Tensor.FromValues([divisor], [1]);
            AccumulateGradient(expanded.Divide(divisorTensor));
        }, this);
    }

    /// <summary>
    /// Inverse of a Sum/Mean reduction along one axis: broadcasts the
    /// (smaller) upstream gradient back up to this variable's original
    /// shape by re-inserting the reduced axis (if it was dropped rather
    /// than kept) and letting <see cref="Tensor.Add"/>'s broadcasting do the
    /// actual repetition.
    /// </summary>
    private Tensor ExpandReducedAxis(Tensor gradient, int axis, bool keepDims)
    {
        var withAxis = gradient;
        if (!keepDims)
        {
            var reshaped = new int[gradient.Shape.Length + 1];
            Array.Copy(gradient.Shape, 0, reshaped, 0, axis);
            reshaped[axis] = 1;
            Array.Copy(gradient.Shape, axis, reshaped, axis + 1, gradient.Shape.Length - axis);
            withAxis = gradient.Reshape(reshaped);
        }

        return Tensor.Zeros(Value.Shape).Add(withAxis);
    }
}
