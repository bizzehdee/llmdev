using System.Diagnostics.CodeAnalysis;
using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

public sealed partial class Tensor
{
    private static Accelerator? _cachedSubtractInPlaceKernelAccelerator;
    private static Action<Index1D, ArrayView<float>, ArrayView<float>>? _cachedSubtractInPlaceKernel;

    /// <summary>
    /// TASK-039: device-resident <see cref="SubtractInPlace"/> - the
    /// actual call <see cref="AdamWOptimizer"/>'s update makes on a
    /// (potentially resident) parameter every step, the specific
    /// remaining contributor TASK-036's measurement pointed at that
    /// TASK-038 didn't cover (that task's scope was `ElementwiseBinary`/
    /// unary ops, not the separate in-place-mutation methods in this
    /// file). Unlike every other GPU op in this codebase, there is no new
    /// output <see cref="Tensor"/> - this mutates <c>this</c> buffer's
    /// existing device memory directly, in place, matching what
    /// <see cref="SubtractInPlace"/> already does for a heap-backed
    /// target. <paramref name="delta"/> doesn't need to already be
    /// resident: if it isn't, its host span is uploaded once as a
    /// transient device buffer - still just one kernel launch and one
    /// upload, not <see cref="Length"/> individual round trips.
    /// </summary>
    private void SubtractInPlaceGpu(GpuFloatBuffer target, Tensor delta)
    {
        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetSubtractInPlaceKernel(accelerator);

        bool deltaIsGpuResident = delta._buffer is GpuFloatBuffer;
        using var transientDelta = deltaIsGpuResident ? null : accelerator.Allocate1D<float>(Length);
        if (!deltaIsGpuResident)
        {
            delta._buffer.TryGetSpan(out var deltaSpan);
            transientDelta!.View.CopyFromCPU(deltaSpan.ToArray());
        }
        var deltaView = deltaIsGpuResident ? ((GpuFloatBuffer)delta._buffer).View : transientDelta!.View;

        kernel(new Index1D(Length), target.View, deltaView);
        accelerator.Synchronize();
    }

    private static Action<Index1D, ArrayView<float>, ArrayView<float>> GetSubtractInPlaceKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedSubtractInPlaceKernelAccelerator, accelerator) || _cachedSubtractInPlaceKernel is null)
        {
            _cachedSubtractInPlaceKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SubtractInPlaceKernel);
            _cachedSubtractInPlaceKernelAccelerator = accelerator;
        }
        return _cachedSubtractInPlaceKernel;
    }

    /// <summary>Excluded from code coverage instrumentation - same confirmed Coverlet/ILGPU interaction as <see cref="MatMulKernel"/>.</summary>
    [ExcludeFromCodeCoverage]
    private static void SubtractInPlaceKernel(Index1D index, ArrayView<float> target, ArrayView<float> delta)
    {
        target[index] -= delta[index];
    }
}
