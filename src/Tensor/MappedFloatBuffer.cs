using Common;

namespace Tensor;

/// <summary>
/// Disk-backed tensor storage for tensors large enough that holding them on
/// the managed heap risks exhausting RAM (see PLAN.md's memory constraint).
/// Thin wrapper around the same <see cref="MappedArray{T}"/> pattern the
/// tokeniser uses for its training state.
/// </summary>
public sealed class MappedFloatBuffer : IFloatBuffer
{
    private readonly MappedArray<float> _data;

    public int Length => _data.Length;

    public MappedFloatBuffer(int length, string scratchDirectory)
    {
        _data = new MappedArray<float>(length, scratchDirectory);
    }

    public float this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    public void Dispose()
    {
        _data.Dispose();
    }

    /// <summary>
    /// Always declines - a deliberate policy choice (PLAN.md/TASK-015),
    /// not a technical limitation: the underlying mapped memory is a
    /// stable native pointer, so a Span over it is technically possible,
    /// but this buffer exists specifically for tensors deliberately routed
    /// to disk because they're too large to want fully "hot" - opting
    /// them into the optimised path's contiguous-access assumption would
    /// undercut that choice. Callers fall back to the scalar path instead.
    /// </summary>
    public bool TryGetSpan(out Span<float> span)
    {
        span = default;
        return false;
    }
}
