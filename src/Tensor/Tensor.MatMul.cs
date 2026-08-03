using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// TASK-021: below this many independent output rows (batchCount * m),
    /// <see cref="Parallel.For(int, int, Action{int})"/>'s thread-scheduling
    /// overhead exceeds the work it would save - e.g. the many `[1]`-shaped
    /// scalar tensors used throughout this codebase for constants, which
    /// have exactly one output row. Chosen conservatively low rather than
    /// tuned against a specific machine: correctness (see the "never
    /// parallelise the inner reduction" note below) doesn't depend on it,
    /// only whether a given call actually benefits from spreading rows
    /// across cores.
    /// </summary>
    private const int MinRowsForParallelMatMul = 64;

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
    /// <see cref="MatMulOptimised"/> for why not every tensor can. Either
    /// way, independent output rows are spread across cores via
    /// <see cref="Parallel"/> once there are enough of them to be worth it
    /// (TASK-021) - see <see cref="MinRowsForParallelMatMul"/>.
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
            && _buffer.TryGetSpan(out _)
            && other._buffer.TryGetSpan(out _))
        {
            return MatMulOptimised(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }

        return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
    }

    /// <summary>
    /// Runs <paramref name="computeRow"/> once for every combined
    /// (batch, output-row) index in [0, totalRows) - sequentially below
    /// <see cref="MinRowsForParallelMatMul"/>, otherwise spread across
    /// cores via <see cref="Parallel.For(int, int, Action{int})"/>.
    /// Deterministic either way: each call writes only the output
    /// element(s) for its own row, so the *set* of writes and their
    /// individual values never depend on how many threads did the work or
    /// in what order - only the untouched inner accumulation loop (the
    /// reduction over k) determines a value's exact floating-point result,
    /// and that stays strictly sequential inside every call.
    /// </summary>
    private static void ForEachRow(int totalRows, Action<int> computeRow)
    {
        if (totalRows >= MinRowsForParallelMatMul)
        {
            Parallel.For(0, totalRows, computeRow);
        }
        else
        {
            for (int row = 0; row < totalRows; row++)
            {
                computeRow(row);
            }
        }
    }

    /// <summary>
    /// Decomposes a flat batch index back into per-dimension coordinates
    /// for <paramref name="shape"/> - the inverse of the flattening
    /// <see cref="Dot(int[], int[])"/> performs with strides. Needed
    /// because parallelising over batch (TASK-021) means a given batch's
    /// coordinates must be computable directly from its flat index, not
    /// incrementally from the previous one the way <see cref="Increment"/>
    /// works (which assumes strictly sequential, single-threaded access).
    /// </summary>
    private static int[] UnravelIndex(int flat, int[] shape)
    {
        var idx = new int[shape.Length];
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            idx[d] = flat % shape[d];
            flat /= shape[d];
        }
        return idx;
    }

    private Tensor MatMulScalar(Tensor other, int[] batchShape, int[] aBatchShape, int[] bBatchShape, int m, int k, int n, int[] outShape)
    {
        var result = Zeros(outShape);

        var aBatchStrides = Strides[..^2];
        var bBatchStrides = other.Strides[..^2];
        int aRowStride = Strides[^2], aColStride = Strides[^1];
        int bRowStride = other.Strides[^2], bColStride = other.Strides[^1];

        int batchCount = Count(batchShape);
        int totalRows = batchCount * m;

        void ComputeRow(int combined)
        {
            int b = combined / m;
            int i = combined % m;
            var batchIdx = UnravelIndex(b, batchShape);
            int aBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, aBatchShape, aBatchStrides);
            int bBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, bBatchShape, bBatchStrides);
            int outBatchOffset = b * m * n;

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

        ForEachRow(totalRows, ComputeRow);

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
    /// This only works with genuinely contiguous spans: <c>this</c>
    /// tensor's data is one directly (every Tensor is contiguous row-major
    /// by construction, so a row - fixed batch/i, varying p - is already a
    /// contiguous run), but <paramref name="other"/>'s *columns* (fixed j,
    /// varying p) are not contiguous in its own row-major layout. Rather
    /// than reading strided elements one at a time (defeating the point),
    /// this transposes <paramref name="other"/> once up front - materialising
    /// a genuinely contiguous copy (see Tensor.Transpose's doc comment)
    /// whose rows are <paramref name="other"/>'s original columns - and
    /// reads contiguous rows from that instead. The transposed copy is
    /// always heap-backed (Transpose calls <see cref="Zeros"/>), regardless
    /// of what backed <paramref name="other"/>, so only <c>this</c> and
    /// <paramref name="other"/> need the heap-backed check the caller
    /// already did.
    ///
    /// Each row re-fetches its spans from <see cref="IFloatBuffer.TryGetSpan"/>
    /// rather than sharing spans captured from the caller (TASK-021): a
    /// <see cref="Span{T}"/> is a ref struct and can't be captured into the
    /// closure <see cref="ForEachRow"/> hands to <see cref="Parallel.For(int, int, Action{int})"/>
    /// - re-deriving it per call is a cheap view over the same underlying
    /// array/pointer, not a copy.
    /// </summary>
    private Tensor MatMulOptimised(Tensor other, int[] batchShape, int[] aBatchShape, int[] bBatchShape, int m, int k, int n, int[] outShape)
    {
        using var otherTransposed = other.Transpose(other.Shape.Length - 2, other.Shape.Length - 1); // [...,n,k]
        if (!otherTransposed._buffer.TryGetSpan(out _))
        {
            // Zeros() (what Transpose produces) is always heap-backed, so
            // this shouldn't happen in practice - fall back defensively
            // rather than assume it never will.
            return MatMulScalar(other, batchShape, aBatchShape, bBatchShape, m, k, n, outShape);
        }

        var result = Zeros(outShape);

        var aBatchStrides = Strides[..^2];
        var bBatchStridesTransposed = otherTransposed.Strides[..^2]; // batch dims untouched by transposing the last two axes
        int aRowStride = Strides[^2];
        int bTransposedRowStride = otherTransposed.Strides[^2]; // = k: each row of the transpose is one of other's original columns

        int batchCount = Count(batchShape);
        int totalRows = batchCount * m;

        void ComputeRow(int combined)
        {
            int b = combined / m;
            int i = combined % m;
            var batchIdx = UnravelIndex(b, batchShape);
            int aBatchOffset = MapBroadcastFlatIndex(batchIdx, batchShape, aBatchShape, aBatchStrides);
            int bBatchOffsetTransposed = MapBroadcastFlatIndex(batchIdx, batchShape, bBatchShape, bBatchStridesTransposed);
            int outBatchOffset = b * m * n;

            _buffer.TryGetSpan(out var aSpan);
            otherTransposed._buffer.TryGetSpan(out var bTransposedSpan);
            result._buffer.TryGetSpan(out var outSpan);

            var aRow = aSpan.Slice(aBatchOffset + i * aRowStride, k);
            for (int j = 0; j < n; j++)
            {
                var bColumnAsRow = bTransposedSpan.Slice(bBatchOffsetTransposed + j * bTransposedRowStride, k);
                outSpan[outBatchOffset + i * n + j] = TensorPrimitives.Dot(aRow, bColumnAsRow);
            }
        }

        ForEachRow(totalRows, ComputeRow);

        return result;
    }
}
