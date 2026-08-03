using System.Diagnostics.CodeAnalysis;
using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

public sealed partial class Tensor
{
    private static Accelerator? _cachedMatMulKernelAccelerator;
    private static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, int, int, int, int, int, int, int>? _cachedMatMulKernel;

    /// <summary>
    /// TASK-032: same math as <see cref="MatMulScalar"/> - deliberately so,
    /// down to reusing the same batch-offset computation
    /// (<see cref="MapBroadcastFlatIndex"/>) - but executed as one ILGPU
    /// kernel launch across every (batch, output-row, output-column)
    /// triple at once, instead of a CPU loop (optionally
    /// <see cref="Parallel"/>-spread, TASK-021) over rows. Unlike
    /// <see cref="MatMulOptimised"/>'s SIMD dot-product approach, a GPU
    /// kernel indexes strided memory directly - the "transpose so the
    /// inner loop reads a contiguous span" trick <see cref="MatMulOptimised"/>
    /// needs for <c>TensorPrimitives.Dot</c> doesn't apply here, so
    /// <paramref name="other"/> is uploaded as-is.
    ///
    /// Always resolves the accelerator via
    /// <see cref="GpuContext.GetAccelerator"/> with CPU fallback allowed -
    /// this method's job is "compute the right answer on whatever
    /// accelerator is available," not "refuse if it's not a real GPU."
    /// That stricter check belongs to a caller deciding *whether to select
    /// <see cref="TensorBackend.Gpu"/> at all* (TASK-033's CLI flag), not
    /// to every individual matmul call once it's already selected.
    ///
    /// TASK-035: an operand that's already device-resident
    /// (<see cref="GpuFloatBuffer"/>, TASK-034) is used directly via its
    /// existing <see cref="GpuFloatBuffer.View"/> instead of being
    /// re-uploaded - the concrete, scoped-down form "chaining" takes in
    /// this codebase today. A caller that keeps a tensor resident across
    /// many calls (a model's weight matrices, reused every forward pass -
    /// TASK-036's job to actually wire that up) stops paying to re-upload
    /// it every single time. The *output* deliberately stays host-backed,
    /// same as before this task: making it device-resident by default
    /// would silently make every *other*, not-yet-GPU-aware op (elementwise
    /// add/multiply, layernorm, activations, reductions) that later touches
    /// that result fall back to <see cref="GpuFloatBuffer"/>'s slow
    /// per-element indexer - correct, but a real, easy-to-miss performance
    /// cliff, and guarding every one of those call sites with an explicit
    /// bulk <see cref="Tensor.ToHost"/> pull-back first is real additional
    /// work this task doesn't attempt. A stated scope decision, not a
    /// discovered gap: full whole-forward-pass on-device chaining remains
    /// TASK-036 (or a further follow-up)'s job, once it's clear which
    /// specific ops besides matmul are worth teaching this.
    /// </summary>
    private Tensor MatMulGpu(Tensor other, int[] batchShape, int[] aBatchShape, int[] bBatchShape, int m, int k, int n, int[] outShape)
    {
        bool aIsGpuResident = _buffer is GpuFloatBuffer;
        bool bIsGpuResident = other._buffer is GpuFloatBuffer;

        Span<float> aSpan = default, bSpan = default;
        if (!aIsGpuResident && !_buffer.TryGetSpan(out aSpan))
        {
            // Zeros()-backed operands always support this; defensive
            // fallback only, mirroring MatMulOptimised's own guard.
            return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }
        if (!bIsGpuResident && !other._buffer.TryGetSpan(out bSpan))
        {
            return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }

        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        var kernel = GetMatMulKernel(accelerator);

        var aBatchStrides = Strides[..^2];
        var bBatchStrides = other.Strides[..^2];
        int aRowStride = Strides[^2], aColStride = Strides[^1];
        int bRowStride = other.Strides[^2], bColStride = other.Strides[^1];

        int batchCount = Count(batchShape);
        var aBatchOffsets = new int[batchCount];
        var bBatchOffsets = new int[batchCount];
        for (int b = 0; b < batchCount; b++)
        {
            var batchIdx = UnravelIndex(b, batchShape);
            aBatchOffsets[b] = MapBroadcastFlatIndex(batchIdx, batchShape, aBatchShape, aBatchStrides);
            bBatchOffsets[b] = MapBroadcastFlatIndex(batchIdx, batchShape, bBatchShape, bBatchStrides);
        }

        // Only allocate (and later dispose) a transient device buffer for
        // an operand that isn't already resident there - an already-resident
        // operand's existing view is used as-is, no re-upload.
        using var transientA = aIsGpuResident ? null : accelerator.Allocate1D<float>(aSpan.Length);
        using var transientB = bIsGpuResident ? null : accelerator.Allocate1D<float>(bSpan.Length);
        transientA?.View.CopyFromCPU(aSpan.ToArray());
        transientB?.View.CopyFromCPU(bSpan.ToArray());
        var aView = aIsGpuResident ? ((GpuFloatBuffer)_buffer).View : transientA!.View;
        var bView = bIsGpuResident ? ((GpuFloatBuffer)other._buffer).View : transientB!.View;

        using var deviceABatchOffsets = accelerator.Allocate1D<int>(batchCount);
        using var deviceBBatchOffsets = accelerator.Allocate1D<int>(batchCount);
        using var deviceOutput = accelerator.Allocate1D<float>(batchCount * m * n);

        deviceABatchOffsets.View.CopyFromCPU(aBatchOffsets);
        deviceBBatchOffsets.View.CopyFromCPU(bBatchOffsets);

        kernel(new Index2D(batchCount * m, n), aView, bView, deviceOutput.View,
            deviceABatchOffsets.View, deviceBBatchOffsets.View,
            m, k, n, aRowStride, aColStride, bRowStride, bColStride);
        accelerator.Synchronize();

        var result = Zeros(outShape);
        result._buffer.TryGetSpan(out var outSpan); // Zeros() is always heap-backed.
        var outputHost = new float[outSpan.Length];
        deviceOutput.View.CopyToCPU(outputHost);
        outputHost.CopyTo(outSpan);

        return result;
    }

    /// <summary>Compiles (once per accelerator instance) and caches <see cref="MatMulKernel"/> - kernel compilation is real, non-trivial cost that shouldn't repeat on every matmul call.</summary>
    private static Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, int, int, int, int, int, int, int> GetMatMulKernel(Accelerator accelerator)
    {
        if (!ReferenceEquals(_cachedMatMulKernelAccelerator, accelerator) || _cachedMatMulKernel is null)
        {
            _cachedMatMulKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>, int, int, int, int, int, int, int>(MatMulKernel);
            _cachedMatMulKernelAccelerator = accelerator;
        }
        return _cachedMatMulKernel;
    }

    /// <summary>
    /// One thread per (combined batch*row, column) output element -
    /// <paramref name="index"/>.X is <c>batch * m + row</c>,
    /// <paramref name="index"/>.Y is the output column. Same accumulation
    /// as <see cref="MatMulScalar"/>'s inner loop, just running on the
    /// accelerator instead of the CPU.
    ///
    /// Excluded from code coverage instrumentation: ILGPU compiles this
    /// method's own IL into a GPU/CPU kernel at runtime
    /// (<see cref="GetMatMulKernel"/>), and Coverlet's source-instrumented
    /// IL is genuinely incompatible with that - not a testing gap, a
    /// confirmed tool interaction (ILGPU's IR importer throws an internal
    /// compiler error against instrumented IL). The method is still fully
    /// exercised by real, passing tests (`dotnet test` without coverage
    /// collection) - only the coverage *measurement* of this one method is
    /// excluded, mirroring this project's existing precedent of excluding
    /// what a tool genuinely cannot measure (e.g. `Program.cs` composition
    /// roots) rather than treating instrumentation as a hard requirement.
    /// </summary>
    [ExcludeFromCodeCoverage]
    private static void MatMulKernel(
        Index2D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> output,
        ArrayView<int> aBatchOffsets,
        ArrayView<int> bBatchOffsets,
        int m, int k, int n,
        int aRowStride, int aColStride,
        int bRowStride, int bColStride)
    {
        int combined = index.X;
        int col = index.Y;
        int batch = combined / m;
        int row = combined % m;

        int aOffset = aBatchOffsets[batch] + row * aRowStride;
        int bOffset = bBatchOffsets[batch];

        float sum = 0f;
        for (int p = 0; p < k; p++)
        {
            sum += a[aOffset + p * aColStride] * b[bOffset + p * bRowStride + col * bColStride];
        }

        output[combined * n + col] = sum;
    }
}
