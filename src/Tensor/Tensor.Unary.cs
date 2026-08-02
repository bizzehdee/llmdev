namespace Tensor;

public sealed partial class Tensor
{
    public Tensor Negate() => ElementwiseUnary(static x => -x);
    public Tensor Exp() => ElementwiseUnary(MathF.Exp);
    public Tensor Log() => ElementwiseUnary(MathF.Log);
    public Tensor Relu() => ElementwiseUnary(static x => MathF.Max(0f, x));
    public Tensor Sqrt() => ElementwiseUnary(MathF.Sqrt);
    public Tensor Scale(float factor) => ElementwiseUnary(x => x * factor);

    /// <summary>1 where the input was positive, 0 elsewhere - the local
    /// derivative of <see cref="Relu"/>, factored out for autodiff's use.</summary>
    public Tensor ReluMask() => ElementwiseUnary(static x => x > 0f ? 1f : 0f);

    private Tensor ElementwiseUnary(Func<float, float> op)
    {
        var result = Zeros(Shape);
        for (int i = 0; i < Length; i++)
        {
            result._buffer[i] = op(_buffer[i]);
        }
        return result;
    }
}
