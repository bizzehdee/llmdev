namespace Tensor;

/// <summary>
/// Flat float storage for a <see cref="Tensor"/>. Abstracted so a tensor's
/// backing memory can be swapped between the managed heap (small tensors)
/// and a disk-backed <see cref="MappedFloatBuffer"/> (large ones) without
/// changing any of the tensor math - see PLAN.md's memory constraint.
/// </summary>
public interface IFloatBuffer : IDisposable
{
    int Length { get; }
    float this[int index] { get; set; }

    /// <summary>
    /// Attempts to get a contiguous span over this buffer's data, for
    /// callers (TASK-015's optimised tensor backend) that need direct
    /// contiguous access for SIMD operations. Only heap-backed storage
    /// supports this (returns true); disk-backed storage declines
    /// (returns false) so large, deliberately-memory-bounded tensors don't
    /// get quietly required to hand out an in-process contiguous view -
    /// see PLAN.md/TASK-015's memory-discipline discussion.
    /// </summary>
    bool TryGetSpan(out Span<float> span);
}
