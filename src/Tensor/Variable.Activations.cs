namespace Tensor;

public sealed partial class Variable
{
    public Variable Exp()
    {
        var value = Value.Exp();
        return FromOp(value, result => () =>
        {
            // d/dx e^x = e^x = result.Value, so reuse the already-computed forward value.
            AccumulateGradient(result.Gradient.Multiply(result.Value));
        }, this);
    }

    public Variable Log()
    {
        var value = Value.Log();
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.Divide(Value));
        }, this);
    }

    public Variable Relu()
    {
        var value = Value.Relu();
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.Multiply(Value.ReluMask()));
        }, this);
    }

    public Variable Sqrt()
    {
        var value = Value.Sqrt();
        return FromOp(value, result => () =>
        {
            // d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/(2*result.Value).
            var two = Tensor.FromValues([2f], [1]);
            AccumulateGradient(result.Gradient.Divide(result.Value.Multiply(two)));
        }, this);
    }

    /// <summary>
    /// Softmax along <paramref name="axis"/>: exp(x) / sum(exp(x), axis).
    /// No max-subtraction numerical-stability trick (see Tensor.Reductions
    /// - there's no Max-along-axis reduction yet); fine at the input scales
    /// this project is working with so far, but a candidate to revisit if
    /// large logits ever cause overflow.
    /// </summary>
    public Variable Softmax(int axis)
    {
        var expValue = Value.Exp();
        var sumValue = expValue.Sum(axis, keepDims: true);
        var value = expValue.Divide(sumValue);

        return FromOp(value, result => () =>
        {
            // Standard softmax-Jacobian-vector product: dx = y * (dy - sum(dy*y, axis)).
            var dyTimesY = result.Gradient.Multiply(result.Value);
            var sumDyTimesY = dyTimesY.Sum(axis, keepDims: true);
            var dx = result.Value.Multiply(result.Gradient.Subtract(sumDyTimesY));
            AccumulateGradient(dx);
        }, this);
    }
}
