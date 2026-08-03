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
    /// Softmax along <paramref name="axis"/>: exp(x - max(x)) / sum(exp(x - max(x)), axis).
    /// Subtracting the per-row max before Exp (TASK-017's "safe softmax"
    /// trick) doesn't change the result mathematically - it cancels in the
    /// ratio - but keeps every Exp argument &lt;= 0, avoiding overflow for
    /// large-magnitude logits. Only affects the forward computation: the
    /// backward pass below is expressed purely in terms of the already-computed
    /// softmax output (result.Value) and its incoming gradient, not the
    /// subtraction itself, so no gradient-formula change is needed.
    /// </summary>
    public Variable Softmax(int axis)
    {
        var maxValue = Value.Max(axis, keepDims: true);
        var shifted = Value.Subtract(maxValue);
        var expValue = shifted.Exp();
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
