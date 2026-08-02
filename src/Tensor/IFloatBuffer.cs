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
}
