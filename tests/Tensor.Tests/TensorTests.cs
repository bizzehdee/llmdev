using Tensor;
using Xunit;

namespace Tensor.Tests;

[Collection("GpuContext")]
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

    // MatMul is tested against all three TensorBackend values throughout
    // (TASK-015's own instruction, extended to TASK-032's Gpu backend: prove
    // correctness by running the *existing* test suite against every backend,
    // not a separate smaller one for the fast path). Tensor.Backend is
    // AsyncLocal-backed specifically so this doesn't leak into other
    // concurrently-running tests - see Tensor.cs's doc comment. The Gpu cases
    // run against whatever GpuContext.GetAccelerator(allowCpuFallback: true)
    // resolves to - ILGPU's CPU accelerator on this machine (confirmed: no
    // working OpenCL driver despite the AMD GPU/ICD registration being
    // present, see GpuContext's own doc comment) - so they prove the kernel's
    // *math* is correct, not that it was actually re-run against real GPU
    // hardware. Anyone with a working CUDA/OpenCL setup should re-run this
    // whole suite once to additionally confirm that (no code change needed -
    // GetAccelerator already prefers a real GPU over CPU whenever one exists).

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    [InlineData(TensorBackend.Gpu)]
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
    [InlineData(TensorBackend.Gpu)]
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
    [InlineData(TensorBackend.Gpu)]
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

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    [InlineData(TensorBackend.Gpu)]
    public void MatMul_ManyRows_UsesParallelPathAndMatchesReferenceImplementation(TensorBackend backend)
    {
        // TASK-021: 100 output rows clears MinRowsForParallelMatMul, so this
        // exercises the Parallel.For path (below the threshold, the earlier
        // MatMul tests already cover the sequential path). Verified against
        // an independent reference implementation, not just "doesn't crash."
        Tensor.Backend = backend;
        try
        {
            const int m = 100, k = 8, n = 5;
            var rng = new Random(42);
            var aValues = new float[m * k];
            var bValues = new float[k * n];
            for (int i = 0; i < aValues.Length; i++) aValues[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bValues.Length; i++) bValues[i] = (float)(rng.NextDouble() * 2 - 1);

            using var a = Tensor.FromValues(aValues, [m, k]);
            using var b = Tensor.FromValues(bValues, [k, n]);

            using var result = a.MatMul(b);

            var expected = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                    {
                        sum += aValues[i * k + p] * bValues[p * n + j];
                    }
                    expected[i * n + j] = sum;
                }
            }

            var actual = result.ToArray();
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], precision: 3);
            }
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Theory]
    [InlineData(TensorBackend.Scalar)]
    [InlineData(TensorBackend.Optimised)]
    [InlineData(TensorBackend.Gpu)]
    public void MatMul_ManyRows_IsDeterministicAcrossRepeatedCalls(TensorBackend backend)
    {
        // Parallelising independent output rows must not make the result
        // depend on scheduling: repeated calls with the same inputs must
        // produce bit-identical output.
        Tensor.Backend = backend;
        try
        {
            const int m = 200, k = 6, n = 4;
            var rng = new Random(7);
            var aValues = new float[m * k];
            var bValues = new float[k * n];
            for (int i = 0; i < aValues.Length; i++) aValues[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < bValues.Length; i++) bValues[i] = (float)(rng.NextDouble() * 2 - 1);

            using var a = Tensor.FromValues(aValues, [m, k]);
            using var b = Tensor.FromValues(bValues, [k, n]);

            using var first = a.MatMul(b);
            using var second = a.MatMul(b);

            Assert.Equal(first.ToArray(), second.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
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
    public void Max_AlongAxis_ReducesAndDropsDimensionByDefault()
    {
        using var t = Tensor.FromValues([1, 5, 3, 9, 2, 6], [2, 3]);

        using var maxRows = t.Max(axis: 1);
        using var maxCols = t.Max(axis: 0);

        Assert.Equal(new[] { 2 }, maxRows.Shape);
        Assert.Equal(new float[] { 5, 9 }, maxRows.ToArray());

        Assert.Equal(new[] { 3 }, maxCols.Shape);
        Assert.Equal(new float[] { 9, 5, 6 }, maxCols.ToArray());
    }

    [Fact]
    public void Max_KeepDims_PreservesRank()
    {
        using var t = Tensor.FromValues([1, 5, 3, 9, 2, 6], [2, 3]);

        using var result = t.Max(axis: 1, keepDims: true);

        Assert.Equal(new[] { 2, 1 }, result.Shape);
        Assert.Equal(new float[] { 5, 9 }, result.ToArray());
    }

    [Fact]
    public void Max_NegativeAxisThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Max(axis: -1));
    }

    [Fact]
    public void Max_AxisTooLargeThrows()
    {
        using var t = Tensor.Zeros([2, 3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => t.Max(axis: 2));
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

    // TASK-034: device-resident storage plumbing. No op yet knows how to
    // consume a device-resident operand without a host round-trip
    // (TASK-035) - existing ops must still decline it and fall back to
    // scalar correctly, the same way they already do for a disk-backed
    // operand, since GpuFloatBuffer.TryGetSpan also always declines.

    [Fact]
    public void ZerosOnGpu_IsGpuResidentAndZeroInitialised()
    {
        using var tensor = Tensor.ZerosOnGpu([2, 2]);

        Assert.True(tensor.IsGpuResident);
        Assert.Equal(new float[] { 0, 0, 0, 0 }, tensor.ToArray());
    }

    [Fact]
    public void ToGpu_ThenToHost_RoundTripsAHeapBackedTensorExactly()
    {
        using var original = Tensor.FromValues([1, 2, 3, 4], [2, 2]);

        using var gpu = original.ToGpu();
        Assert.True(gpu.IsGpuResident);

        using var backOnHost = gpu.ToHost();
        Assert.False(backOnHost.IsGpuResident);
        Assert.Equal(original.ToArray(), backOnHost.ToArray());
    }

    [Fact]
    public void ToGpu_ThenToHost_RoundTripsADiskBackedTensorExactly()
    {
        using var disk = Tensor.ZerosOnDisk([2, 2], ScratchDirectory);
        disk[0, 0] = 5; disk[0, 1] = 6; disk[1, 0] = 7; disk[1, 1] = 8;

        using var gpu = disk.ToGpu();
        Assert.True(gpu.IsGpuResident);

        using var backOnHost = gpu.ToHost();
        Assert.Equal(new float[] { 5, 6, 7, 8 }, backOnHost.ToArray());
    }

    [Fact]
    public void ToGpu_OnAnAlreadyGpuResidentTensor_ReturnsTheSameInstance()
    {
        using var gpu = Tensor.ZerosOnGpu([2, 2]);

        var again = gpu.ToGpu();

        Assert.Same(gpu, again);
    }

    [Fact]
    public void ToHost_OnAHeapBackedTensor_ReturnsTheSameInstance()
    {
        using var heap = Tensor.FromValues([1, 2, 3, 4], [2, 2]);

        var again = heap.ToHost();

        Assert.Same(heap, again);
    }

    // TASK-035: matmul uses an already-resident operand's existing device
    // view directly instead of re-uploading it - correctness must hold
    // whichever operand (or both, or neither) is already GPU-resident.

    [Fact]
    public void MatMul_OneOperandGpuResident_StaysCorrectUnderGpuBackend()
    {
        Tensor.Backend = TensorBackend.Gpu;
        try
        {
            using var heap = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
            using var gpuResident = Tensor.FromValues([5, 6, 7, 8], [2, 2]).ToGpu();

            using var result = heap.MatMul(gpuResident);

            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void MatMul_OtherOperandGpuResident_StaysCorrectUnderGpuBackend()
    {
        Tensor.Backend = TensorBackend.Gpu;
        try
        {
            using var gpuResident = Tensor.FromValues([1, 2, 3, 4], [2, 2]).ToGpu();
            using var heap = Tensor.FromValues([5, 6, 7, 8], [2, 2]);

            using var result = gpuResident.MatMul(heap);

            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void MatMul_BothOperandsGpuResident_StaysCorrectUnderGpuBackend()
    {
        Tensor.Backend = TensorBackend.Gpu;
        try
        {
            using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]).ToGpu();
            using var b = Tensor.FromValues([5, 6, 7, 8], [2, 2]).ToGpu();

            using var result = a.MatMul(b);

            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void MatMul_SameGpuResidentOperandReusedAcrossManyCalls_StaysCorrectEveryTime()
    {
        // The actual point of TASK-035: a weight-like tensor kept resident
        // once and reused across many matmul calls (as a real training
        // loop would reuse a model's parameters every forward pass)
        // mustn't be re-uploaded or otherwise corrupted by repeated use.
        Tensor.Backend = TensorBackend.Gpu;
        try
        {
            using var weight = Tensor.FromValues([1, 0, 0, 1], [2, 2]).ToGpu(); // identity
            using var input1 = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
            using var input2 = Tensor.FromValues([5, 6, 7, 8], [2, 2]);

            using var result1 = input1.MatMul(weight);
            using var result2 = input2.MatMul(weight);
            using var result3 = input1.MatMul(weight);

            Assert.Equal(input1.ToArray(), result1.ToArray());
            Assert.Equal(input2.ToArray(), result2.ToArray());
            Assert.Equal(input1.ToArray(), result3.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void MatMul_DiskBackedOperandUnderGpuBackend_FallsBackToScalarAndStaysCorrect()
    {
        // Unlike a GPU-resident operand (now handled directly, above), a
        // disk-backed one still can't be used by the GPU path - same
        // TryGetSpan-declines gate as before this task.
        Tensor.Backend = TensorBackend.Gpu;
        try
        {
            using var heap = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
            using var disk = Tensor.ZerosOnDisk([2, 2], ScratchDirectory);
            disk[0, 0] = 5; disk[0, 1] = 6; disk[1, 0] = 7; disk[1, 1] = 8;

            using var result = heap.MatMul(disk);

            Assert.Equal(new float[] { 19, 22, 43, 50 }, result.ToArray());
        }
        finally
        {
            Tensor.Backend = TensorBackend.Scalar;
        }
    }

    [Fact]
    public void Concat_AlongAxisZero_AppendsRows()
    {
        using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var b = Tensor.FromValues([5, 6], [1, 2]);

        using var result = a.Concat(b, axis: 0);

        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, result.ToArray());
    }

    [Fact]
    public void Concat_AlongAxisOne_AppendsColumns()
    {
        using var a = Tensor.FromValues([1, 2, 3, 4], [2, 2]);
        using var b = Tensor.FromValues([5, 6], [2, 1]);

        using var result = a.Concat(b, axis: 1);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(new float[] { 1, 2, 5, 3, 4, 6 }, result.ToArray());
    }

    [Fact]
    public void Concat_ThreeDimensional_AppendsAlongMiddleAxis()
    {
        // The shape KV-cache growth actually uses: [numHeads, seqLen, headDim].
        using var a = Tensor.FromValues([1, 2, 3, 4], [2, 1, 2]);
        using var b = Tensor.FromValues([5, 6, 7, 8], [2, 1, 2]);

        using var result = a.Concat(b, axis: 1);

        Assert.Equal(new[] { 2, 2, 2 }, result.Shape);
        Assert.Equal(new float[] { 1, 2, 5, 6, 3, 4, 7, 8 }, result.ToArray());
    }

    [Fact]
    public void Concat_RankMismatchThrows()
    {
        using var a = Tensor.Zeros([2, 2]);
        using var b = Tensor.Zeros([2, 2, 1]);

        Assert.Throws<InvalidOperationException>(() => a.Concat(b, axis: 0));
    }

    [Fact]
    public void Concat_ShapeMismatchOutsideAxisThrows()
    {
        using var a = Tensor.Zeros([2, 3]);
        using var b = Tensor.Zeros([5, 3]);

        Assert.Throws<InvalidOperationException>(() => a.Concat(b, axis: 1));
    }

    [Fact]
    public void Concat_AxisOutOfRangeThrows()
    {
        using var a = Tensor.Zeros([2, 2]);
        using var b = Tensor.Zeros([2, 2]);

        Assert.Throws<ArgumentOutOfRangeException>(() => a.Concat(b, axis: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Concat(b, axis: 2));
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
