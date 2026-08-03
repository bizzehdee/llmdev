namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Concatenates <c>this</c> and <paramref name="other"/> along
    /// <paramref name="axis"/> - every other dimension must already match.
    /// No gradient (this is a plain <see cref="Tensor"/> op, not a
    /// <see cref="Variable"/> one): added for TASK-020's KV-cache, which
    /// grows a cached Key/Value tensor by appending newly computed
    /// positions onto it every generation step, entirely at inference
    /// time - there's no backward pass to support there.
    /// </summary>
    public Tensor Concat(Tensor other, int axis)
    {
        if (Shape.Length != other.Shape.Length)
        {
            throw new InvalidOperationException($"Concat requires tensors of the same rank: {Shape.Length} vs {other.Shape.Length}.");
        }
        if (axis < 0 || axis >= Shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis must be in [0,{Shape.Length}) for shape [{string.Join(",", Shape)}].");
        }
        for (int d = 0; d < Shape.Length; d++)
        {
            if (d != axis && Shape[d] != other.Shape[d])
            {
                throw new InvalidOperationException($"Shapes must match outside the concat axis: [{string.Join(",", Shape)}] vs [{string.Join(",", other.Shape)}].");
            }
        }

        var outShape = (int[])Shape.Clone();
        outShape[axis] = Shape[axis] + other.Shape[axis];
        var result = Zeros(outShape);

        CopyInto(this, result, axis, destAxisOffset: 0);
        CopyInto(other, result, axis, destAxisOffset: Shape[axis]);

        return result;
    }

    private static void CopyInto(Tensor src, Tensor dest, int axis, int destAxisOffset)
    {
        var idx = new int[src.Shape.Length];
        for (int n = 0; n < src.Length; n++)
        {
            int destFlat = Dot(idx, dest.Strides) + destAxisOffset * dest.Strides[axis];
            dest._buffer[destFlat] = src._buffer[n];
            Increment(idx, src.Shape);
        }
    }
}
