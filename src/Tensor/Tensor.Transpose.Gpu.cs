using System.Diagnostics.CodeAnalysis;
using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

public sealed partial class Tensor
{
    private static Accelerator? _cachedTransposeKernelAccelerator;
    private static Action<Index2D, ArrayView<float>, ArrayView<float>, int, int>? _cachedTransposeKernel;

    /// <summary>
    /// TASK-037: swaps the last two dimensions of an already device-resident
    /// tensor via one ILGPU kernel launch, producing a device-resident
    /// result - no host round-trip either way, unlike
    /// <see cref="TransposeScalar"/>'s per-element buffer indexing (correct
    /// against a <see cref="GpuFloatBuffer"/> too, just a real host↔device
    /// round trip per element). Always resolves the accelerator via
    /// <see cref="GpuContext.GetAccelerator"/> with CPU fallback allowed -
    /// same policy as <see cref="MatMulGpu"/>: this method's job is
    /// "transpose correctly on whatever accelerator this tensor's data
    /// already lives on," not to gate on real GPU hardware itself.
    /// </summary>
    private Tensor TransposeGpu()
    {
        int m = Shape[^2];
        int n = Shape[^1];
        int batchCount = Length / (m * n);

        var newShape = (int[])Shape.Clone();
        (newShape[^2], newShape[^1]) = (newShape[^1], newShape[^2]);

        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetTransposeKernel(accelerator);

        var inputView = ((GpuFloatBuffer)_buffer).View;
        var outputBuffer = new GpuFloatBuffer(Length);

        kernel(new Index2D(batchCount * n, m), inputView, outputBuffer.View, m, n);
        accelerator.Synchronize();

        return new Tensor(outputBuffer, newShape);
    }

    /// <summary>Compiles (once per accelerator instance) and caches <see cref="TransposeLastTwoDimsKernel"/> - same reasoning as <see cref="GetMatMulKernel"/>.</summary>
    private static Action<Index2D, ArrayView<float>, ArrayView<float>, int, int> GetTransposeKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedTransposeKernelAccelerator, accelerator) || _cachedTransposeKernel is null)
        {
            _cachedTransposeKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, int, int>(TransposeLastTwoDimsKernel);
            _cachedTransposeKernelAccelerator = accelerator;
        }
        return _cachedTransposeKernel;
    }

    /// <summary>
    /// One thread per (combined batch*n + i, j) output element, where the
    /// output's last two dimensions are [n, m] (swapped from the input's
    /// [m, n]): <c>output[batch, i, j] = input[batch, j, i]</c>. Both
    /// input and output are flat, contiguous, row-major within each batch.
    ///
    /// Excluded from code coverage instrumentation for the same confirmed
    /// reason as <see cref="MatMulKernel"/>: ILGPU compiles this method's
    /// own IL at runtime, and Coverlet's instrumented IL is incompatible
    /// with that - not a testing gap, a confirmed tool interaction. Fully
    /// exercised by real, passing tests without coverage collection.
    /// </summary>
    [ExcludeFromCodeCoverage]
    private static void TransposeLastTwoDimsKernel(Index2D index, ArrayView<float> input, ArrayView<float> output, int m, int n)
    {
        int combined = index.X; // batch * n + i
        int j = index.Y;
        int batch = combined / n;
        int i = combined % n;

        int inputFlat = batch * (m * n) + j * n + i;
        int outputFlat = combined * m + j;

        output[outputFlat] = input[inputFlat];
    }
}
