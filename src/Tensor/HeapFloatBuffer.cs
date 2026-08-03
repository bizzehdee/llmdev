namespace Tensor;

/// <summary>Plain managed-heap backing for small tensors.</summary>
public sealed class HeapFloatBuffer : IFloatBuffer
{
    private readonly float[] _data;

    public int Length => _data.Length;

    public HeapFloatBuffer(int length)
    {
        _data = new float[length];
    }

    public float this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    public void Dispose()
    {
        // Nothing to release; the GC reclaims _data normally.
    }

    public bool TryGetSpan(out Span<float> span)
    {
        span = _data;
        return true;
    }
}
