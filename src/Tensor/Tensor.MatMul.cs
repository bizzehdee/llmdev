namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Matrix multiplication over the last two dimensions of each operand
    /// (shape [..., m, k] x [..., k, n] -> [..., m, n]); any leading "batch"
    /// dimensions are broadcast against each other the same way elementwise
    /// ops broadcast (so a plain 2D matrix multiplies against every matrix
    /// in a batch, batches of equal shape multiply position-for-position,
    /// etc.) - this is what multi-head attention needs later: a batch of
    /// (seq, head_dim) matrices per head, per sequence in the batch.
    /// </summary>
    public Tensor MatMul(Tensor other)
    {
        if (Shape.Length < 2 || other.Shape.Length < 2)
        {
            throw new InvalidOperationException("MatMul requires tensors of rank >= 2.");
        }

        int m = Shape[^2];
        int k = Shape[^1];
        int k2 = other.Shape[^2];
        int n = other.Shape[^1];
        if (k != k2)
        {
            throw new InvalidOperationException($"Inner dimensions must match for matmul: {k} vs {k2}.");
        }

        var aBatchShape = Shape[..^2];
        var bBatchShape = other.Shape[..^2];
        var batchShape = BroadcastShape(aBatchShape, bBatchShape);
        var aBatchStrides = Strides[..^2];
        var bBatchStrides = other.Strides[..^2];

        var outShape = new int[batchShape.Length + 2];
        batchShape.CopyTo(outShape, 0);
        outShape[^2] = m;
        outShape[^1] = n;
        var result = Zeros(outShape);

        int aRowStride = Strides[^2], aColStride = Strides[^1];
        int bRowStride = other.Strides[^2], bColStride = other.Strides[^1];

        var batchIdx = new int[batchShape.Length];
        int batchCount = Count(batchShape);
        for (int b = 0; b < batchCount; b++)
        {
            int aBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, aBatchShape, aBatchStrides);
            int bBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, bBatchShape, bBatchStrides);
            int outBatchOffset = b * m * n;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                    {
                        float aVal = _buffer[aBatchOffset + i * aRowStride + p * aColStride];
                        float bVal = other._buffer[bBatchOffset + p * bRowStride + j * bColStride];
                        sum += aVal * bVal;
                    }
                    result._buffer[outBatchOffset + i * n + j] = sum;
                }
            }

            Increment(batchIdx, batchShape);
        }

        return result;
    }
}
