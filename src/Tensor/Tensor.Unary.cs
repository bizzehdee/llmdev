namespace Tensor;

public sealed partial class Tensor
{
    public Tensor Negate() => ElementwiseUnary(static x => -x);
    public Tensor Exp() => ElementwiseUnary(MathF.Exp);
    public Tensor Log() => ElementwiseUnary(MathF.Log);
    public Tensor Relu() => ElementwiseUnary(static x => MathF.Max(0f, x));

    /// <summary>
    /// TASK-038: uses the device-resident kernel path when this tensor is
    /// already GPU-resident - <see cref="AdamWOptimizer"/>'s per-parameter
    /// update calls this directly on (potentially resident) moment
    /// estimates. Not <see cref="Backend"/>-gated, same reasoning as
    /// <see cref="Transpose"/>/<see cref="Add"/>. Every other unary op
    /// (<see cref="Negate"/>, <see cref="Exp"/>, <see cref="Log"/>,
    /// <see cref="Relu"/>, <see cref="ReluMask"/>) stays on the general
    /// scalar path - deliberately not brought on-device in this task,
    /// since nothing on the resident-weights path calls them.
    /// </summary>
    public Tensor Sqrt() => _buffer is GpuFloatBuffer ? SqrtGpu() : ElementwiseUnary(MathF.Sqrt);

    /// <summary>TASK-038: same device-resident dispatch as <see cref="Sqrt"/> - <see cref="AdamWOptimizer"/>'s update calls this directly on the (potentially resident) parameter and moment estimates.</summary>
    public Tensor Scale(float factor) => _buffer is GpuFloatBuffer ? ScaleGpu(factor) : ElementwiseUnary(x => x * factor);

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
