namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Swaps two dimensions, materialising a new contiguous tensor (this
    /// type has no strided-view support - see the class doc comment on
    /// Tensor.cs - so this is a full copy, not a zero-cost reshape).
    ///
    /// TASK-037: dispatches to a device-resident kernel
    /// (<see cref="TransposeGpu"/>) when this tensor is already
    /// GPU-resident (<see cref="GpuFloatBuffer"/>) *and* the swap is of
    /// the last two dimensions specifically - the one shape
    /// <see cref="Variable.MatMul"/>'s backward pass actually calls this
    /// with (transposing a weight to compute the other operand's
    /// gradient), the biggest measured contributor to TASK-036's ≈37×
    /// slowdown. Every other case (any other dimension pair, or a
    /// heap/disk-backed operand) uses the unchanged general scalar path -
    /// this isn't a <see cref="TensorBackend"/>-gated choice the way
    /// <see cref="MatMul"/>'s is, since there's no competing
    /// implementation to prefer here: a resident tensor should always get
    /// the efficient path when one exists for its shape, regardless of
    /// whatever <see cref="Backend"/> happens to be selected for
    /// unrelated ops at that moment.
    /// </summary>
    public Tensor Transpose(int dim0, int dim1)
    {
        if (dim0 < 0 || dim0 >= Shape.Length || dim1 < 0 || dim1 >= Shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(dim0), $"Dimensions must be in [0,{Shape.Length}) for shape [{string.Join(",", Shape)}].");
        }

        bool isLastTwoDims = Shape.Length >= 2 &&
            ((dim0 == Shape.Length - 2 && dim1 == Shape.Length - 1) || (dim0 == Shape.Length - 1 && dim1 == Shape.Length - 2));

        if (isLastTwoDims && _buffer is GpuFloatBuffer)
        {
            return TransposeGpu();
        }

        return TransposeScalar(dim0, dim1);
    }

    private Tensor TransposeScalar(int dim0, int dim1)
    {
        var newShape = (int[])Shape.Clone();
        (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);
        var result = Zeros(newShape);

        var idx = new int[Shape.Length];
        var dstIdx = new int[Shape.Length];
        for (int n = 0; n < Length; n++)
        {
            idx.CopyTo(dstIdx, 0);
            (dstIdx[dim0], dstIdx[dim1]) = (dstIdx[dim1], dstIdx[dim0]);

            result._buffer[Dot(dstIdx, result.Strides)] = _buffer[n];
            Increment(idx, Shape);
        }

        return result;
    }
}
