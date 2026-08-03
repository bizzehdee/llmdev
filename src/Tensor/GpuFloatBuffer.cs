using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

/// <summary>
/// TASK-034: device-resident tensor storage - a third <see cref="IFloatBuffer"/>
/// kind alongside <see cref="HeapFloatBuffer"/>/<see cref="MappedFloatBuffer"/>,
/// backed by an ILGPU device memory buffer whose lifetime spans the
/// <see cref="Tensor"/> that owns it, not one op call. This is the storage
/// piece only - TASK-032's <c>MatMulGpu</c> still allocates its own
/// transient device buffers per call and copies back to a heap-backed
/// result immediately, so a tensor built on this buffer isn't yet
/// consumed any faster by matmul than a heap-backed one is (TASK-035
/// teaches ops to chain on a buffer like this one without a host
/// round-trip in between - that's where the actual performance this line
/// of work is chasing comes from).
///
/// Always resolves its accelerator via <see cref="GpuContext.GetAccelerator"/>
/// with CPU fallback allowed - like <c>MatMulGpu</c>, this buffer's job is
/// "hold data on whatever accelerator is available," not "refuse if it's
/// not a real GPU" (that stricter check belongs to a caller choosing
/// <see cref="TensorBackend.Gpu"/> in the first place, i.e. the CLI flag).
/// </summary>
public sealed class GpuFloatBuffer : IFloatBuffer
{
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _data;

    public int Length { get; }

    /// <summary>
    /// The underlying device view - TASK-035's escape hatch for an op
    /// (currently just <c>MatMulGpu</c>) that wants to use this buffer's
    /// data directly on the accelerator instead of copying it to a
    /// transient device buffer first, when this buffer is already
    /// resident there. Internal: nothing outside <c>Tensor</c>'s own
    /// GPU-op implementations should reach into a buffer's raw device
    /// storage.
    /// </summary>
    internal ArrayView1D<float, Stride1D.Dense> View => _data.View;

    public GpuFloatBuffer(int length)
    {
        Length = length;
        var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);
        _data = accelerator.Allocate1D<float>(Math.Max(length, 1));
        // ILGPU doesn't zero-initialize a fresh allocation - HeapFloatBuffer's
        // `new float[length]` and MappedFloatBuffer's fresh scratch file both
        // do, so this buffer matches that guarantee explicitly rather than
        // silently starting with whatever was already in that device memory.
        _data.View.MemSetToZero();
    }

    /// <summary>
    /// Per-element access to device memory - a real host↔device round
    /// trip *every call*, deliberately not fast. This exists for
    /// correctness (tests, small constant tensors, debugging), not for
    /// hot-path use; anything performance-sensitive should use
    /// <see cref="CopyFromHost"/>/<see cref="CopyToHost"/> for one bulk
    /// transfer, or (once TASK-035 lands) an op that consumes this buffer
    /// without ever touching the host at all.
    /// </summary>
    public float this[int index]
    {
        get
        {
            var host = new float[1];
            _data.View.SubView(index, 1).CopyToCPU(host);
            return host[0];
        }
        set
        {
            _data.View.SubView(index, 1).CopyFromCPU([value]);
        }
    }

    /// <summary>One bulk host→device transfer for this buffer's entire contents - the efficient alternative to setting every index individually.</summary>
    public void CopyFromHost(ReadOnlySpan<float> source)
    {
        _data.View.CopyFromCPU(source.ToArray());
    }

    /// <summary>One bulk device→host transfer for this buffer's entire contents - the efficient alternative to reading every index individually.</summary>
    public void CopyToHost(Span<float> destination)
    {
        var host = new float[Length];
        _data.View.CopyToCPU(host);
        host.CopyTo(destination);
    }

    public void Dispose()
    {
        _data.Dispose();
    }

    /// <summary>
    /// Always declines, the same policy as <see cref="MappedFloatBuffer"/>
    /// and for a related reason: there is no host-addressable span over
    /// device memory, so callers (e.g. <c>MatMulOptimised</c>'s SIMD path)
    /// must fall back to the scalar path for a device-resident operand,
    /// same as they already do for a disk-backed one.
    /// </summary>
    public bool TryGetSpan(out Span<float> span)
    {
        span = default;
        return false;
    }
}
