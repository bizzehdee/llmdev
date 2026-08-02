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
}
