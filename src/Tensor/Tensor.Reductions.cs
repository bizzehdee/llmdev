namespace Tensor;

public sealed partial class Tensor
{
    public Tensor Sum(int axis, bool keepDims = false)
    {
        if (axis < 0 || axis >= Shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis must be in [0,{Shape.Length}) for shape [{string.Join(",", Shape)}].");
        }

        var outShape = keepDims
            ? Shape.Select((dim, i) => i == axis ? 1 : dim).ToArray()
            : Shape.Where((_, i) => i != axis).ToArray();
        var result = Zeros(outShape);

        var idx = new int[Shape.Length];
        for (int n = 0; n < Length; n++)
        {
            int[] dstIdx = keepDims
                ? idx.Select((v, i) => i == axis ? 0 : v).ToArray()
                : idx.Where((_, i) => i != axis).ToArray();

            int dstFlat = Dot(dstIdx, result.Strides);
            result._buffer[dstFlat] += _buffer[n];
            Increment(idx, Shape);
        }

        return result;
    }

    public Tensor Mean(int axis, bool keepDims = false)
    {
        var summed = Sum(axis, keepDims);
        float divisor = Shape[axis];
        for (int i = 0; i < summed.Length; i++)
        {
            summed._buffer[i] /= divisor;
        }
        return summed;
    }
}
