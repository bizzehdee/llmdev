namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Swaps two dimensions, materialising a new contiguous tensor (this
    /// type has no strided-view support - see the class doc comment on
    /// Tensor.cs - so this is a full copy, not a zero-cost reshape).
    /// </summary>
    public Tensor Transpose(int dim0, int dim1)
    {
        if (dim0 < 0 || dim0 >= Shape.Length || dim1 < 0 || dim1 >= Shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(dim0), $"Dimensions must be in [0,{Shape.Length}) for shape [{string.Join(",", Shape)}].");
        }

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
