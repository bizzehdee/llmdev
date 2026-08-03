using System.Diagnostics.CodeAnalysis;
using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

/// <summary>Which elementwise binary op <see cref="Tensor.ElementwiseBinaryKernel"/> performs - an int opcode, not a delegate, because an ILGPU kernel can't invoke an arbitrary host closure.</summary>
internal enum ElementwiseOp
{
    Add,
    Subtract,
    Multiply,
    Divide,
}

public sealed partial class Tensor
{
    private static Accelerator? _cachedElementwiseKernelAccelerator;
    private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>? _cachedElementwiseBinaryKernel;
    private static Action<Index1D, ArrayView<float>, ArrayView<float>, float>? _cachedScaleKernel;
    private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _cachedSqrtKernel;

    /// <summary>
    /// TASK-038: same-shape elementwise binary op, executed as one ILGPU
    /// kernel launch over both device-resident operands, producing a
    /// device-resident result - no broadcasting support (see
    /// <see cref="ElementwiseBinaryDispatch"/>'s doc comment for why
    /// that's a deliberate, narrower scope than the general scalar path).
    /// </summary>
    private Tensor ElementwiseBinaryGpu(Tensor other, ElementwiseOp opCode)
    {
        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetElementwiseBinaryKernel(accelerator);

        var aView = ((GpuFloatBuffer)_buffer).View;
        var bView = ((GpuFloatBuffer)other._buffer).View;
        var outputBuffer = new GpuFloatBuffer(Length);

        kernel(new Index1D(Length), aView, bView, outputBuffer.View, (int)opCode);
        accelerator.Synchronize();

        return new Tensor(outputBuffer, Shape);
    }

    /// <summary>TASK-038: device-resident <see cref="Scale"/> - used directly by <see cref="AdamWOptimizer"/>'s per-parameter update.</summary>
    private Tensor ScaleGpu(float factor)
    {
        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetScaleKernel(accelerator);

        var inputView = ((GpuFloatBuffer)_buffer).View;
        var outputBuffer = new GpuFloatBuffer(Length);

        kernel(new Index1D(Length), inputView, outputBuffer.View, factor);
        accelerator.Synchronize();

        return new Tensor(outputBuffer, Shape);
    }

    /// <summary>TASK-038: device-resident <see cref="Sqrt"/> - used directly by <see cref="AdamWOptimizer"/>'s per-parameter update.</summary>
    private Tensor SqrtGpu()
    {
        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetSqrtKernel(accelerator);

        var inputView = ((GpuFloatBuffer)_buffer).View;
        var outputBuffer = new GpuFloatBuffer(Length);

        kernel(new Index1D(Length), inputView, outputBuffer.View);
        accelerator.Synchronize();

        return new Tensor(outputBuffer, Shape);
    }

    private static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> GetElementwiseBinaryKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedElementwiseKernelAccelerator, accelerator) || _cachedElementwiseBinaryKernel is null)
        {
            _cachedElementwiseBinaryKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(ElementwiseBinaryKernel);
            _cachedElementwiseKernelAccelerator = accelerator;
        }
        return _cachedElementwiseBinaryKernel;
    }

    private static Action<Index1D, ArrayView<float>, ArrayView<float>, float> GetScaleKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedElementwiseKernelAccelerator, accelerator) || _cachedScaleKernel is null)
        {
            _cachedScaleKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(ScaleKernel);
            _cachedElementwiseKernelAccelerator = accelerator;
        }
        return _cachedScaleKernel;
    }

    private static Action<Index1D, ArrayView<float>, ArrayView<float>> GetSqrtKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedElementwiseKernelAccelerator, accelerator) || _cachedSqrtKernel is null)
        {
            _cachedSqrtKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SqrtKernel);
            _cachedElementwiseKernelAccelerator = accelerator;
        }
        return _cachedSqrtKernel;
    }

    /// <summary>
    /// One thread per output element; <paramref name="opCode"/> picks the
    /// operation via a device-side branch (an <see cref="ElementwiseOp"/>
    /// int, not a delegate - a kernel can't invoke an arbitrary host
    /// closure). Excluded from code coverage instrumentation for the same
    /// confirmed reason as <see cref="MatMulKernel"/>/
    /// <see cref="TransposeLastTwoDimsKernel"/>: ILGPU compiles this
    /// method's own IL at runtime, and Coverlet's instrumented IL is
    /// incompatible with that. Fully exercised by real, passing tests
    /// without coverage collection.
    /// </summary>
    [ExcludeFromCodeCoverage]
    private static void ElementwiseBinaryKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> output, int opCode)
    {
        output[index] = (ElementwiseOp)opCode switch
        {
            ElementwiseOp.Add => a[index] + b[index],
            ElementwiseOp.Subtract => a[index] - b[index],
            ElementwiseOp.Multiply => a[index] * b[index],
            ElementwiseOp.Divide => a[index] / b[index],
            _ => 0f,
        };
    }

    /// <summary>Excluded from code coverage instrumentation - see <see cref="ElementwiseBinaryKernel"/>'s doc comment.</summary>
    [ExcludeFromCodeCoverage]
    private static void ScaleKernel(Index1D index, ArrayView<float> input, ArrayView<float> output, float factor)
    {
        output[index] = input[index] * factor;
    }

    /// <summary>Excluded from code coverage instrumentation - see <see cref="ElementwiseBinaryKernel"/>'s doc comment.</summary>
    [ExcludeFromCodeCoverage]
    private static void SqrtKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        output[index] = MathF.Sqrt(input[index]);
    }
}
