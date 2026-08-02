namespace Tensor;

public sealed partial class Tensor
{
    public Tensor Add(Tensor other) => ElementwiseBinary(other, static (a, b) => a + b);
    public Tensor Subtract(Tensor other) => ElementwiseBinary(other, static (a, b) => a - b);
    public Tensor Multiply(Tensor other) => ElementwiseBinary(other, static (a, b) => a * b);
    public Tensor Divide(Tensor other) => ElementwiseBinary(other, static (a, b) => a / b);

    private Tensor ElementwiseBinary(Tensor other, Func<float, float, float> op)
    {
        var outShape = BroadcastShape(Shape, other.Shape);
        var result = Zeros(outShape);

        var idx = new int[outShape.Length];
        int total = Count(outShape);
        for (int n = 0; n < total; n++)
        {
            int aFlat = MapBroadcastFlatIndex(idx, outShape, Shape, Strides);
            int bFlat = MapBroadcastFlatIndex(idx, outShape, other.Shape, other.Strides);
            result._buffer[n] = op(_buffer[aFlat], other._buffer[bFlat]);
            Increment(idx, outShape);
        }

        return result;
    }

    /// <summary>
    /// NumPy-style broadcasting: shapes are aligned from the right, and any
    /// pair of dimensions must either match or have one of them equal to 1.
    /// Missing leading dimensions on the shorter shape are treated as 1.
    /// </summary>
    private static int[] BroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            int db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (da != db && da != 1 && db != 1)
            {
                throw new InvalidOperationException($"Shapes [{string.Join(",", a)}] and [{string.Join(",", b)}] are not broadcastable.");
            }
            result[i] = Math.Max(da, db);
        }
        return result;
    }

    /// <summary>
    /// Maps a coordinate in the broadcast output shape back to a flat index
    /// into a (possibly lower-rank, possibly size-1-dimensioned) source
    /// tensor: size-1 and padded leading dimensions contribute 0 (i.e. the
    /// same source element is reused for every broadcast position along
    /// that axis).
    /// </summary>
    private static int MapBroadcastFlatIndex(int[] outIdx, int[] outShape, int[] srcShape, int[] srcStrides)
    {
        int rankDiff = outShape.Length - srcShape.Length;
        int flat = 0;
        for (int i = rankDiff; i < outShape.Length; i++)
        {
            int j = i - rankDiff;
            int coord = srcShape[j] == 1 ? 0 : outIdx[i];
            flat += coord * srcStrides[j];
        }
        return flat;
    }
}
