namespace Tensor;

public sealed partial class Variable
{
    public Variable Reshape(int[] newShape)
    {
        var value = Value.Reshape(newShape);
        return FromOp(value, result => () =>
        {
            // Reshape doesn't change element correspondence, just the shape
            // labels on top of the same row-major data, so its backward is
            // just reshaping the gradient back to the original shape.
            AccumulateGradient(result.Gradient.Reshape(Value.Shape));
        }, this);
    }
}
