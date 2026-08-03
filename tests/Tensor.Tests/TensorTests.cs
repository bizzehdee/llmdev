using Tensor;
using Xunit;

namespace Tensor.Tests;

public class TensorTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "tensor-tests-scratch");

    static TensorTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void Zeros_HasCorrectShapeStridesAndAllZeroValues()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Equal(new[] { 2, 3 }, t.Shape);
        Assert.Equal(new[] { 3, 1 }, t.Strides);
        Assert.Equal(6, t.Length);
        Assert.All(t.ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void FromValues_PopulatesInRowMajorOrder()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        Assert.Equal(1f, t[0, 0]);
        Assert.Equal(2f, t[0, 1]);
        Assert.Equal(3f, t[0, 2]);
        Assert.Equal(4f, t[1, 0]);
        Assert.Equal(6f, t[1, 2]);
    }

    [Fact]
    public void FromValues_WrongCountThrows()
    {
        Assert.Throws<ArgumentException>(() => Tensor.FromValues([1, 2, 3], [2, 2]));
    }

    [Fact]
    public void Indexer_GetSetRoundtrips()
    {
        using var t = Tensor.Zeros([2, 2]);

        t[1, 0] = 42f;

        Assert.Equal(42f, t[1, 0]);
        Assert.Equal(0f, t[0, 0]);
    }

    [Fact]
    public void Indexer_OutOfRangeThrows()
    {
        using var t = Tensor.Zeros([2, 2]);

        Assert.Throws<IndexOutOfRangeException>(() => t[2, 0]);
    }

    [Fact]
    public void Indexer_NegativeIndexThrows()
    {
        using var t = Tensor.Zeros([2, 2]);

        Assert.Throws<IndexOutOfRangeException>(() => t[-1, 0]);
    }

    [Fact]
    public void Indexer_WrongRankThrows()
    {
        using var t = Tensor.Zeros([2, 2]);

        Assert.Throws<ArgumentException>(() => t[0]);
    }

    [Fact]
    public void Add_SameShape_AddsElementwise()
    {
        using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var b = Tensor.FromValues([10, 20, 30, 40], [2, 2]);

        using var result = a.Add(b);

        Assert.Equal(new float[] { 11, 22, 33, 44 }, result.ToArray());
    }

    [Fact]
    public void Subtract_SameShape_SubtractsElementwise()
    {
        using var a = Tensor.FromValues([10, 20, 30, 40], [2, 2]);
        using var b = Tensor.FromValues([1, 2, 3, 4], [2, 2]);

        using var result = a.Subtract(b);

        Assert.Equal(new float[] { 9, 18, 27, 36 }, result.ToArray());
    }

    [Fact]
    public void Multiply_SameShape_MultipliesElementwise()
    {
        using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var b = Tensor.FromValues([2, 2, 2, 2], [2, 2]);

        using var result = a.Multiply(b);

        Assert.Equal(new float[] { 2, 4, 6, 8 }, result.ToArray());
    }

    [Fact]
    public void Divide_SameShape_DividesElementwise()
    {
        using var a = Tensor.FromValues([10, 20, 30, 40], [2, 2]);
        using var b = Tensor.FromValues([2, 2, 2, 2], [2, 2]);

        using var result = a.Divide(b);

        Assert.Equal(new float[] { 5, 10, 15, 20 }, result.ToArray());
    }

    [Fact]
    public void Add_BroadcastsScalarAgainstMatrix()
    {
        using var matrix = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var scalar = Tensor.FromValues([10], [1]);

        using var result = matrix.Add(scalar);

        Assert.Equal(new float[] { 11, 12, 13, 14 }, result.ToArray());
    }

    [Fact]
    public void Add_BroadcastsMatrixAgainstScalar_LowerRankOperandFirst()
    {
        // Same broadcast as above, but with the lower-rank operand as the
        // receiver (`this`) rather than the argument - exercises the "pad
        // this tensor's missing leading dimensions" side of broadcasting,
        // not just "pad the argument's".
        using var scalar = Tensor.FromValues([10], [1]);
        using var matrix = Tensor.FromValues([1, 2, 3, 4], [2, 2]);

        using var result = scalar.Add(matrix);

        Assert.Equal(new float[] { 11, 12, 13, 14 }, result.ToArray());
    }

    [Fact]
    public void Add_BroadcastsRowVectorAgainstMatrix()
    {
        using var matrix = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);
        using var row = Tensor.FromValues([10, 20, 30], [3]);

        using var result = matrix.Add(row);

        Assert.Equal(new float[] { 11, 22, 33, 14, 25, 36 }, result.ToArray());
    }

    [Fact]
    public void Add_BroadcastsColumnVectorAgainstMatrix()
    {
        using var matrix = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);
        using var column = Tensor.FromValues([100, 200], [2, 1]);

        using var result = matrix.Add(column);

        Assert.Equal(new float[] { 101, 102, 103, 204, 205, 206 }, result.ToArray());
    }

    [Fact]
    public void Add_IncompatibleShapesThrows()
    {
        using var a = Tensor.Zeros([2, 3]);
        using var b = Tensor.Zeros([2, 4]);

        Assert.Throws<InvalidOperationException>(() => a.Add(b));
    }

    // MatMul is tested against both TensorBackend values throughout (TASK-015's
    // own instruction: prove correctness by running the *existing* test suite
    // against both backends, not a separate smaller one for the fast path).
    // Tensor.Backend is AsyncLocal-backed specifically so this doesn't leak
    // into other concurrently-running tests - see Tensor.cs's doc comment.

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    public void MatMul_TwoByTwoKnownResult(TensorBackend backend)
    {
        Tensor.Backend = backend;
        try
        {
            // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
            using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
            using var b = Tensor.FromValues([5, 6, 7, 8], [2, 2]);

            using var result = a.MatMul(b);

            Assert.Equal(new[] { 2, 2 }, result.Shape);
            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    public void MatMul_NonSquareShapes(TensorBackend backend)
    {
        Tensor.Backend = backend;
        try
        {
            // (2x3) x (3x2) -> (2x2)
            using var a = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);
            using var b = Tensor.FromValues([7, 8, 9, 10, 11, 12], [3, 2]);

            using var result = a.MatMul(b);

            // row0 . col0 = 1*7+2*9+3*11 = 58 ; row0 . col1 = 1*8+2*10+3*12 = 64
            // row1 . col0 = 4*7+5*9+6*11 = 139; row1 . col1 = 4*8+5*10+6*12 = 154
            Assert.Equal(new[] { 2, 2 }, result.Shape);
            Assert.Equal(new float[] { 58, 64, 139, 154 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    public void MatMul_BatchedAgainstSingleMatrixBroadcasts(TensorBackend backend)
    {
        Tensor.Backend = backend;
        try
        {
            // batch of two (2x2) matrices x one (2x2) matrix -> batch of two (2x2) results
            using var batch = Tensor.FromValues([1, 2, 3, 4, 1, 0, 0, 1], [2, 2, 2]);
            using var shared = Tensor.FromValues([5, 6, 7, 8], [2, 2]);

            using var result = batch.MatMul(shared);

            Assert.Equal(new[] { 2, 2, 2 }, result.Shape);
            // batch[0] = [[1,2],[3,4]] x shared = [[19,22],[43,50]]
            // batch[1] = identity x shared = shared = [[5,6],[7,8]]
            Assert.Equal(new float[] { 19, 22, 43, 50, 5, 6, 7, 8 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void MatMul_MismatchedInnerDimensionThrows()
    {
        using var a = Tensor.Zeros([2, 3]);
        using var b = Tensor.Zeros([4, 2]);

        Assert.Throws<InvalidOperationException>(() => a.MatMul(b));
    }

    [Fact]
    public void MatMul_ThisRankBelowTwoThrows()
    {
        using var a = Tensor.Zeros([3]);
        using var b = Tensor.Zeros([3, 2]);

        Assert.Throws<InvalidOperationException>(() => a.MatMul(b));
    }

    [Fact]
    public void MatMul_OtherRankBelowTwoThrows()
    {
        using var a = Tensor.Zeros([2, 3]);
        using var b = Tensor.Zeros([3]);

        Assert.Throws<InvalidOperationException>(() => a.MatMul(b));
    }

    [Fact]
    public void Transpose_TwoDimensional_SwapsRowsAndColumns()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.Transpose(0, 1);

        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(new float[] { 1, 4, 2, 5, 3, 6 }, result.ToArray());
    }

    [Fact]
    public void Transpose_IsSelfInverse()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var roundTripped = t.Transpose(0, 1).Transpose(0, 1);

        Assert.Equal(t.Shape, roundTripped.Shape);
        Assert.Equal(t.ToArray(), roundTripped.ToArray());
    }

    [Fact]
    public void Transpose_NegativeDim0Throws()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Transpose(-1, 0));
    }

    [Fact]
    public void Transpose_Dim0TooLargeThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Transpose(2, 0));
    }

    [Fact]
    public void Transpose_NegativeDim1Throws()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Transpose(0, -1));
    }

    [Fact]
    public void Transpose_Dim1TooLargeThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Transpose(0, 2));
    }

    [Fact]
    public void Sum_AlongAxis_ReducesAndDropsDimensionByDefault()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var summedRows = t.Sum(axis: 1);
        using var summedCols = t.Sum(axis: 0);

        Assert.Equal(new[] { 2 }, summedRows.Shape);
        Assert.Equal(new float[] { 6, 15 }, summedRows.ToArray());

        Assert.Equal(new[] { 3 }, summedCols.Shape);
        Assert.Equal(new float[] { 5, 7, 9 }, summedCols.ToArray());
    }

    [Fact]
    public void Sum_KeepDims_PreservesRank()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.Sum(axis: 1, keepDims: true);

        Assert.Equal(new[] { 2, 1 }, result.Shape);
        Assert.Equal(new float[] { 6, 15 }, result.ToArray());
    }

    [Fact]
    public void Mean_AlongAxis_AveragesElements()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.Mean(axis: 1);

        Assert.Equal(new float[] { 2, 5 }, result.ToArray());
    }

    [Fact]
    public void Sum_NegativeAxisThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Sum(axis: -1));
    }

    [Fact]
    public void Sum_AxisTooLargeThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Sum(axis: 2));
    }

    [Fact]
    public void DiskBackedTensor_BehavesIdenticallyToHeapBacked()
    {
        using var heap = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var disk = Tensor.ZerosOnDisk([2, 2], ScratchDirectory);

        disk[0, 0] = 1;
        disk[0, 1] = 2;
        disk[1, 0] = 3;
        disk[1, 1] = 4;

        using var result = heap.Add(disk);

        Assert.Equal(new float[] { 2, 4, 6, 8 }, result.ToArray());
    }

    [Fact]
    public void MatMul_DiskBackedOperand_FallsBackToScalarAndStaysCorrectUnderOptimisedBackend()
    {
        // TASK-015: a disk-backed operand must decline the optimised path
        // (MappedFloatBuffer.TryGetSpan always returns false) and still
        // produce the correct result via the scalar fallback, even though
        // Backend is set to Optimised.
        Tensor.Backend = TensorBackend.Optimised;
        try
        {
            using var heap = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
            using var disk = Tensor.ZerosOnDisk([2, 2], ScratchDirectory);
            disk[0, 0] = 5;
            disk[0, 1] = 6;
            disk[1, 0] = 7;
            disk[1, 1] = 8;

            using var result = heap.MatMul(disk);

            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void Reshape_PreservesDataInRowMajorOrder()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.Reshape([3, 2]);

        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, result.ToArray());
    }

    [Fact]
    public void Reshape_WrongElementCountThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentException>(() => t.Reshape([4, 2]));
    }

    [Fact]
    public void SumTo_ReducesExtraLeadingDimensions()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.SumTo([3]);

        Assert.Equal(new[] { 3 }, result.Shape);
        Assert.Equal(new float[] { 5, 7, 9 }, result.ToArray());
    }

    [Fact]
    public void SumTo_ReducesBroadcastSizeOneDimensions()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.SumTo([2, 1]);

        Assert.Equal(new[] { 2, 1 }, result.Shape);
        Assert.Equal(new float[] { 6, 15 }, result.ToArray());
    }

    [Fact]
    public void SumTo_MatchingShapeIsUnchanged()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4], [2, 2]);

        using var result = t.SumTo([2, 2]);

        Assert.Equal(t.ToArray(), result.ToArray());
    }

    [Fact]
    public void Negate_FlipsSign()
    {
        using var t = Tensor.FromValues([1, -2, 3], [3]);

        using var result = t.Negate();

        Assert.Equal(new float[] { -1, 2, -3 }, result.ToArray());
    }

    [Fact]
    public void Exp_MatchesMathExp()
    {
        using var t = Tensor.FromValues([0, 1, 2], [3]);

        using var result = t.Exp();

        Assert.Equal(new[] { 1f, MathF.E, MathF.E * MathF.E }, result.ToArray(), EqualityComparer());
    }

    [Fact]
    public void Log_IsInverseOfExp()
    {
        using var t = Tensor.FromValues([1, 2, 3], [3]);

        using var result = t.Exp().Log();

        Assert.Equal(t.ToArray(), result.ToArray(), EqualityComparer());
    }

    [Fact]
    public void Relu_ZeroesNegativesAndPassesPositivesThrough()
    {
        using var t = Tensor.FromValues([-2, -0.5f, 0, 1, 3], [5]);

        using var result = t.Relu();

        Assert.Equal(new float[] { 0, 0, 0, 1, 3 }, result.ToArray());
    }

    [Fact]
    public void ReluMask_IsOneWherePositiveElseZero()
    {
        using var t = Tensor.FromValues([-2, 0, 3], [3]);

        using var result = t.ReluMask();

        Assert.Equal(new float[] { 0, 0, 1 }, result.ToArray());
    }

    [Fact]
    public void GatherRows_SelectsRowsByIndex()
    {
        using var t = Tensor.FromValues([1, 2, 10, 20, 100, 200], [3, 2]);

        using var result = t.GatherRows([2, 0, 0]);

        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(new float[] { 100, 200, 1, 2, 1, 2 }, result.ToArray());
    }

    [Fact]
    public void GatherRows_OutOfRangeIndexThrows()
    {
        using var t = Tensor.Zeros([3, 2]);

        Assert.Throws<IndexOutOfRangeException>(() => t.GatherRows([5]));
    }

    [Fact]
    public void GatherRows_NegativeIndexThrows()
    {
        using var t = Tensor.Zeros([3, 2]);

        Assert.Throws<IndexOutOfRangeException>(() => t.GatherRows([-1]));
    }

    [Fact]
    public void GatherRows_WrongRankThrows()
    {
        using var t = Tensor.Zeros([3]);

        Assert.Throws<InvalidOperationException>(() => t.GatherRows([0]));
    }

    [Fact]
    public void ScatterAddRows_WrongRankThrows()
    {
        using var t = Tensor.Zeros([3]);

        Assert.Throws<InvalidOperationException>(() => t.ScatterAddRows([0], targetRowCount: 1));
    }

    [Fact]
    public void ScatterAddRows_MismatchedIndexCountThrows()
    {
        using var t = Tensor.Zeros([3, 2]);

        Assert.Throws<InvalidOperationException>(() => t.ScatterAddRows([0, 1], targetRowCount: 5));
    }

    [Fact]
    public void ScatterAddRows_AccumulatesRepeatedIndices()
    {
        using var grad = Tensor.FromValues([1, 1, 2, 2, 3, 3], [3, 2]);

        // Rows 0 and 2 both scatter to target row 0; row 1 scatters to target row 1.
        using var result = grad.ScatterAddRows([0, 1, 0], targetRowCount: 2);

        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(new float[] { 4, 4, 2, 2 }, result.ToArray());
    }

    [Fact]
    public void ScatterAddRows_IsGatherRowsAdjoint()
    {
        // Sanity check the pairing: gathering then scattering back with the
        // same indices onto a zero tensor of the original row count should
        // reproduce the gathered rows added into their original positions.
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [3, 2]);

        using var gathered = t.GatherRows([1, 2]);
        using var scattered = gathered.ScatterAddRows([1, 2], targetRowCount: 3);

        Assert.Equal(new float[] { 0, 0, 3, 4, 5, 6 }, scattered.ToArray());
    }

    [Fact]
    public void GatherColumns_PicksOneElementPerRow()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var result = t.GatherColumns([2, 0]);

        Assert.Equal(new[] { 2 }, result.Shape);
        Assert.Equal(new float[] { 3, 4 }, result.ToArray());
    }

    [Fact]
    public void GatherColumns_OutOfRangeColumnThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<IndexOutOfRangeException>(() => t.GatherColumns([3, 0]));
    }

    [Fact]
    public void GatherColumns_WrongIndexCountThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<InvalidOperationException>(() => t.GatherColumns([0, 1, 2]));
    }

    [Fact]
    public void GatherColumns_WrongRankThrows()
    {
        using var t = Tensor.Zeros([3]);

        Assert.Throws<InvalidOperationException>(() => t.GatherColumns([0]));
    }

    [Fact]
    public void GatherColumns_NegativeColumnThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<IndexOutOfRangeException>(() => t.GatherColumns([-1, 0]));
    }

    [Fact]
    public void ScatterAddColumns_WrongRankThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<InvalidOperationException>(() => t.ScatterAddColumns([0, 1], columnCount: 3));
    }

    [Fact]
    public void ScatterAddColumns_MismatchedIndexCountThrows()
    {
        using var t = Tensor.Zeros([2]);

        Assert.Throws<InvalidOperationException>(() => t.ScatterAddColumns([0, 1, 2], columnCount: 3));
    }

    [Fact]
    public void ScatterAddColumns_PlacesEachValueAtItsColumnIndex()
    {
        using var t = Tensor.FromValues([10, 20], [2]);

        using var result = t.ScatterAddColumns([2, 0], columnCount: 3);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(new float[] { 0, 0, 10, 20, 0, 0 }, result.ToArray());
    }

    [Fact]
    public void ScatterAddColumns_IsGatherColumnsAdjoint()
    {
        using var t = Tensor.FromValues([1, 2, 3, 4, 5, 6], [2, 3]);

        using var gathered = t.GatherColumns([1, 2]);
        using var scattered = gathered.ScatterAddColumns([1, 2], columnCount: 3);

        Assert.Equal(new float[] { 0, 2, 0, 0, 0, 6 }, scattered.ToArray());
    }

    [Fact]
    public void Scale_MultipliesEveryElementByFactor()
    {
        using var t = Tensor.FromValues([1, 2, 3], [3]);

        using var result = t.Scale(2.5f);

        Assert.Equal(new float[] { 2.5f, 5f, 7.5f }, result.ToArray());
    }

    [Fact]
    public void SubtractInPlace_MutatesBufferDirectly()
    {
        using var t = Tensor.FromValues([10, 20, 30], [3]);
        using var delta = Tensor.FromValues([1, 2, 3], [3]);

        t.SubtractInPlace(delta);

        Assert.Equal(new float[] { 9, 18, 27 }, t.ToArray());
    }

    [Fact]
    public void SubtractInPlace_ShapeMismatchThrows()
    {
        using var t = Tensor.Zeros([2, 2]);
        using var delta = Tensor.Zeros([4]);

        Assert.Throws<InvalidOperationException>(() => t.SubtractInPlace(delta));
    }

    [Fact]
    public void LoadInPlace_OverwritesBufferDirectly()
    {
        using var t = Tensor.Zeros([3]);

        t.LoadInPlace([1, 2, 3]);

        Assert.Equal(new float[] { 1, 2, 3 }, t.ToArray());
    }

    [Fact]
    public void LoadInPlace_WrongLengthThrows()
    {
        using var t = Tensor.Zeros([3]);

        Assert.Throws<InvalidOperationException>(() => t.LoadInPlace([1, 2]));
    }

    private static IEqualityComparer<float> EqualityComparer() => new ApproximateFloatComparer(1e-5f);

    private sealed class ApproximateFloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => MathF.Abs(x - y) <= tolerance;
        public int GetHashCode(float obj) => 0;
    }
}
