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

    /// <summary>
    /// Max along <paramref name="axis"/>. Used by TASK-017's "safe softmax"
    /// trick (subtract the per-row max before Exp so large logits can't
    /// overflow) - no backward pass needed for that use, since the max is
    /// subtracted before the differentiable Exp/Sum/Divide chain runs, not
    /// woven into it.
    /// </summary>
    public Tensor Max(int axis, bool keepDims = false)
    {
        if (axis < 0 || axis >= Shape.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis must be in [0,{Shape.Length}) for shape [{string.Join(",", Shape)}].");
        }

        var outShape = keepDims
            ? Shape.Select((dim, i) => i == axis ? 1 : dim).ToArray()
            : Shape.Where((_, i) => i != axis).ToArray();
        var result = Zeros(outShape);
        for (int i = 0; i < result.Length; i++)
        {
            result._buffer[i] = float.NegativeInfinity;
        }

        var idx = new int[Shape.Length];
        for (int n = 0; n < Length; n++)
        {
            int[] dstIdx = keepDims
                ? idx.Select((v, i) => i == axis ? 0 : v).ToArray()
                : idx.Where((_, i) => i != axis).ToArray();

            int dstFlat = Dot(dstIdx, result.Strides);
            if (_buffer[n] > result._buffer[dstFlat])
            {
                result._buffer[dstFlat] = _buffer[n];
            }
            Increment(idx, Shape);
        }

        return result;
    }
}
