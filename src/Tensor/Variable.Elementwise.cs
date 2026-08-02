namespace Tensor;

public sealed partial class Variable
{
    public static Variable operator +(Variable a, Variable b) => a.Add(b);
    public static Variable operator -(Variable a, Variable b) => a.Subtract(b);
    public static Variable operator *(Variable a, Variable b) => a.Multiply(b);
    public static Variable operator /(Variable a, Variable b) => a.Divide(b);
    public static Variable operator -(Variable a) => a.Negate();

    public Variable Add(Variable other)
    {
        var value = Value.Add(other.Value);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.SumTo(Value.Shape));
            other.AccumulateGradient(result.Gradient.SumTo(other.Value.Shape));
        }, this, other);
    }

    public Variable Subtract(Variable other)
    {
        var value = Value.Subtract(other.Value);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.SumTo(Value.Shape));
            other.AccumulateGradient(result.Gradient.Negate().SumTo(other.Value.Shape));
        }, this, other);
    }

    public Variable Multiply(Variable other)
    {
        var value = Value.Multiply(other.Value);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.Multiply(other.Value).SumTo(Value.Shape));
            other.AccumulateGradient(result.Gradient.Multiply(Value).SumTo(other.Value.Shape));
        }, this, other);
    }

    public Variable Divide(Variable other)
    {
        var value = Value.Divide(other.Value);
        return FromOp(value, result => () =>
        {
            // d(a/b)/da = 1/b ; d(a/b)/db = -a/b^2
            AccumulateGradient(result.Gradient.Divide(other.Value).SumTo(Value.Shape));
            var dOther = result.Gradient.Multiply(Value).Negate().Divide(other.Value.Multiply(other.Value));
            other.AccumulateGradient(dOther.SumTo(other.Value.Shape));
        }, this, other);
    }

    public Variable Negate()
    {
        var value = Value.Negate();
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.Negate());
        }, this);
    }
}
