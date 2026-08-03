namespace Tensor;

/// <summary>
/// An N-dimensional float array. Row-major (C-order), contiguous storage
/// only - no strided views - which keeps indexing and every op below
/// simple to reason about at the cost of an extra copy for things like
/// <see cref="Transpose"/> that a view-based design could avoid. Storage is
/// swappable between the managed heap (<see cref="Zeros"/>) and a
/// disk-backed scratch file (<see cref="ZerosOnDisk"/>) per PLAN.md's
/// memory constraint; ops in the other Tensor.*.cs files always produce
/// heap-backed results, leaving the choice of when a *result* should be
/// disk-backed to the caller (later stages, once tensor sizes are large
/// enough to matter).
/// </summary>
public sealed partial class Tensor : IDisposable
{
    private readonly IFloatBuffer _buffer;

    private static readonly AsyncLocal<TensorBackend> _backend = new();

    /// <summary>
    /// Which implementation hot ops (matmul) use for any Tensor code
    /// running in the current async/logical-call context. Backed by
    /// <see cref="AsyncLocal{T}"/> rather than a plain static field
    /// deliberately: a plain static would leak between concurrently
    /// running xUnit tests (and, once TASK-021 adds real parallelism,
    /// between unrelated concurrent work in general) - AsyncLocal gives
    /// each independent call chain its own value, defaulting to
    /// <see cref="TensorBackend.Scalar"/> (0) if never set. A CLI sets
    /// this once near startup (see Chat's --optimised flag) for
    /// everything downstream of that point in the same call chain.
    /// </summary>
    public static TensorBackend Backend
    {
        get => _backend.Value;
        set => _backend.Value = value;
    }

    public int[] Shape { get; }
    public int[] Strides { get; }
    public int Length { get; }

    private Tensor(IFloatBuffer buffer, int[] shape)
    {
        Shape = shape;
        Strides = ComputeStrides(shape);
        Length = Count(shape);
        _buffer = buffer;
    }

    public static Tensor Zeros(int[] shape) => new(new HeapFloatBuffer(Count(shape)), shape);

    public static Tensor ZerosOnDisk(int[] shape, string scratchDirectory) =>
        new(new MappedFloatBuffer(Count(shape), scratchDirectory), shape);

    public static Tensor FromValues(float[] values, int[] shape)
    {
        int expected = Count(shape);
        if (values.Length != expected)
        {
            throw new ArgumentException($"Shape [{string.Join(",", shape)}] expects {expected} values, got {values.Length}.");
        }

        var tensor = Zeros(shape);
        for (int i = 0; i < values.Length; i++)
        {
            tensor._buffer[i] = values[i];
        }
        return tensor;
    }

    public float this[params int[] indices]
    {
        get => _buffer[FlatIndex(indices)];
        set => _buffer[FlatIndex(indices)] = value;
    }

    public float[] ToArray()
    {
        var result = new float[Length];
        for (int i = 0; i < Length; i++)
        {
            result[i] = _buffer[i];
        }
        return result;
    }

    public void Dispose() => _buffer.Dispose();

    private int FlatIndex(int[] indices)
    {
        if (indices.Length != Shape.Length)
        {
            throw new ArgumentException($"Expected {Shape.Length} indices for shape [{string.Join(",", Shape)}], got {indices.Length}.");
        }

        int flat = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= Shape[i])
            {
                throw new IndexOutOfRangeException($"Index {indices[i]} out of range for dimension {i} of size {Shape[i]}.");
            }
            flat += indices[i] * Strides[i];
        }
        return flat;
    }

    private static int Count(int[] shape)
    {
        int count = 1;
        foreach (int dim in shape)
        {
            count *= dim;
        }
        return count;
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    private static int Dot(int[] coords, int[] strides)
    {
        int flat = 0;
        for (int i = 0; i < coords.Length; i++)
        {
            flat += coords[i] * strides[i];
        }
        return flat;
    }

    /// <summary>
    /// Increments a row-major "odometer" index in place: rightmost
    /// dimension fastest, matching the flattening order every op below
    /// relies on. Returns false once it wraps back to all zeros (i.e. every
    /// position has been visited).
    /// </summary>
    private static bool Increment(int[] idx, int[] shape)
    {
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            idx[d]++;
            if (idx[d] < shape[d])
            {
                return true;
            }
            idx[d] = 0;
        }
        return false;
    }
}
