namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Reinterprets this tensor's data under a new shape of the same total
    /// element count. Since storage is always contiguous row-major, this is
    /// conceptually free, but is implemented as a copy for now (simplicity
    /// over the added bookkeeping of shared, ref-counted buffers).
    /// </summary>
    public Tensor Reshape(int[] newShape)
    {
        int newCount = Count(newShape);
        if (newCount != Length)
        {
            throw new ArgumentException($"Cannot reshape [{string.Join(",", Shape)}] ({Length} elements) to [{string.Join(",", newShape)}] ({newCount} elements).");
        }

        var result = Zeros(newShape);
        for (int i = 0; i < Length; i++)
        {
            result._buffer[i] = _buffer[i];
        }
        return result;
    }

    /// <summary>
    /// The inverse of broadcasting: sums this tensor down to
    /// <paramref name="targetShape"/>, which must be a shape this tensor
    /// could itself have been broadcast from (every dimension either
    /// matches or is 1, and <paramref name="targetShape"/> may have fewer
    /// leading dimensions). This is exactly what a broadcast op's gradient
    /// needs - e.g. adding a [3] bias to a [2,3] tensor means the bias's
    /// gradient must be summed back down from [2,3] to [3].
    /// </summary>
    public Tensor SumTo(int[] targetShape)
    {
        var result = this;

        int rankDiff = result.Shape.Length - targetShape.Length;
        for (int i = 0; i < rankDiff; i++)
        {
            result = result.Sum(axis: 0, keepDims: false);
        }

        for (int i = 0; i < targetShape.Length; i++)
        {
            if (targetShape[i] == 1 && result.Shape[i] != 1)
            {
                result = result.Sum(axis: i, keepDims: true);
            }
        }

        return result;
    }
}
