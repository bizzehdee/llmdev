namespace Tensor;

public sealed partial class Variable
{
    /// <summary>
    /// out = a @ b, shapes [...,m,k] x [...,k,n] -> [...,m,n]. Standard
    /// matmul backward: da = dOut @ b^T, db = a^T @ dOut, where ^T
    /// transposes only the last two dimensions (the matrix dims, not any
    /// batch dims). If a or b's batch dimensions were broadcast during the
    /// forward pass, the raw da/db computed this way come out at the
    /// *broadcast* batch shape, so SumTo brings them back down to the
    /// original operand shape - same idea as the elementwise ops.
    /// </summary>
    public Variable MatMul(Variable other)
    {
        var value = Value.MatMul(other.Value);
        return FromOp(value, result => () =>
        {
            var bT = other.Value.Transpose(other.Value.Shape.Length - 2, other.Value.Shape.Length - 1);
            var aT = Value.Transpose(Value.Shape.Length - 2, Value.Shape.Length - 1);

            var dA = result.Gradient.MatMul(bT);
            var dB = aT.MatMul(result.Gradient);

            AccumulateGradient(dA.SumTo(Value.Shape));
            other.AccumulateGradient(dB.SumTo(other.Value.Shape));
        }, this, other);
    }
}
