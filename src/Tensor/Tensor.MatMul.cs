using System.Numerics.Tensors;

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
    ///
    /// Uses the <see cref="Backend"/>-selected implementation: the scalar
    /// triple loop by default, or (opt-in, see TASK-015) a
    /// TensorPrimitives-backed dot product for the inner reduction when
    /// both operands can hand out a contiguous span - see
    /// <see cref="MatMulOptimised"/> for why not every tensor can.
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

        var outShape = new int[batchShape.Length + 2];
        batchShape.CopyTo(outShape, 0);
        outShape[^2] = m;
        outShape[^1] = n;

        if (Backend == TensorBackend.Optimised
            && _buffer.TryGetSpan(out var aSpan)
            && other._buffer.TryGetSpan(out _))
        {
            return MatMulOptimised(other, aSpan, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }

        return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
    }

    private Tensor MatMulScalar(Tensor other, int[] batchShape, int[] aBatchShape, int[] bBatchShape, int m, int k, int n, int[] outShape)
    {
        var result = Zeros(outShape);

        var aBatchStrides = Strides[..^2];
        var bBatchStrides = other.Strides[..^2];
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

    /// <summary>
    /// Same math as <see cref="MatMulScalar"/>, but the inner k-length
    /// reduction for each output element is done via
    /// <c>TensorPrimitives.Dot</c> (SIMD) instead of a scalar accumulation
    /// loop - the one op TASK-015 targets, since matmul's O(k) reduction
    /// per output element, repeated m*n*batchCount times, is the dominant
    /// cost in a transformer forward/backward pass.
    ///
    /// This only works with genuinely contiguous spans: <paramref name="aSpan"/>
    /// covers <c>this</c> tensor's data directly (every Tensor is
    /// contiguous row-major by construction, so a row - fixed batch/i,
    /// varying p - is already a contiguous run), but <paramref name="other"/>'s
    /// *columns* (fixed j, varying p) are not contiguous in its own
    /// row-major layout. Rather than reading strided elements one at a
    /// time (defeating the point), this transposes <paramref name="other"/>
    /// once up front - materialising a genuinely contiguous copy (see
    /// Tensor.Transpose's doc comment) whose rows are <paramref name="other"/>'s
    /// original columns - and reads contiguous rows from that instead.
    /// The transposed copy is always heap-backed (Transpose calls
    /// <see cref="Zeros"/>), regardless of what backed <paramref name="other"/>,
    /// so only <c>this</c> and <paramref name="other"/> need the
    /// heap-backed check the caller already did.
    /// </summary>
    private Tensor MatMulOptimised(Tensor other, ReadOnlySpan<float> aSpan, int[] batchShape, int[] aBatchShape, int[] bBatchShape, int m, int k, int n, int[] outShape)
    {
        using var otherTransposed = other.Transpose(other.Shape.Length - 2, other.Shape.Length - 1); // [...,n,k]
        if (!otherTransposed._buffer.TryGetSpan(out var bTransposedSpan))
        {
            // Zeros() (what Transpose produces) is always heap-backed, so
            // this shouldn't happen in practice - fall back defensively
            // rather than assume it never will.
            return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }

        var result = Zeros(outShape);
        result._buffer.TryGetSpan(out var outSpan); // always succeeds: Zeros() is heap-backed.

        var aBatchStrides = Strides[..^2];
        var bBatchStridesTransposed = otherTransposed.Strides[..^2]; // batch dims untouched by transposing the last two axes
        int aRowStride = Strides[^2];
        int bTransposedRowStride = otherTransposed.Strides[^2]; // = k: each row of the transpose is one of other's original columns

        var batchIdx = new int[batchShape.Length];
        int batchCount = Count(batchShape);
        for (int b = 0; b < batchCount; b++)
        {
            int aBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, aBatchShape, aBatchStrides);
            int bBatchOffsetTransposed = MapBroadcastFlatIndex(batchIdx, batchShape, bBatchShape, bBatchStridesTransposed);
            int outBatchOffset = b * m * n;

            for (int i = 0; i < m; i++)
            {
                var aRow = aSpan.Slice(aBatchOffset + i * aRowStride, k);
                for (int j = 0; j < n; j++)
                {
                    var bColumnAsRow = bTransposedSpan.Slice(bBatchOffsetTransposed + j * bTransposedRowStride, k);
                    outSpan[outBatchOffset + i * n + j] = TensorPrimitives.Dot(aRow, bColumnAsRow);
                }
            }

            Increment(batchIdx, batchShape);
        }

        return result;
    }
}
