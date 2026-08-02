namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Selects rows by index from a rank-2 tensor - the core op an
    /// embedding lookup needs (token id -> its row in the embedding
    /// table). <paramref name="rowIndices"/> may repeat.
    /// </summary>
    public Tensor GatherRows(int[] rowIndices)
    {
        if (Shape.Length != 2)
        {
            throw new InvalidOperationException($"GatherRows requires a rank-2 tensor, got rank {Shape.Length}.");
        }

        int cols = Shape[1];
        var result = Zeros([rowIndices.Length, cols]);
        for (int r = 0; r < rowIndices.Length; r++)
        {
            int srcRow = rowIndices[r];
            if (srcRow < 0 || srcRow >= Shape[0])
            {
                throw new IndexOutOfRangeException($"Row index {srcRow} out of range for {Shape[0]} rows.");
            }
            for (int c = 0; c < cols; c++)
            {
                result._buffer[r * cols + c] = _buffer[srcRow * Strides[0] + c * Strides[1]];
            }
        }
        return result;
    }

    /// <summary>
    /// The inverse of <see cref="GatherRows"/> for backpropagation: scatters
    /// (accumulating, i.e. summing rather than overwriting on repeats) each
    /// row of this tensor into <paramref name="rowIndices"/>[row] of a new
    /// [<paramref name="targetRowCount"/>, cols] tensor. Accumulation on
    /// repeats matters because a token appearing twice in a lookup gets
    /// gradient contributions from both occurrences.
    /// </summary>
    public Tensor ScatterAddRows(int[] rowIndices, int targetRowCount)
    {
        if (Shape.Length != 2 || Shape[0] != rowIndices.Length)
        {
            throw new InvalidOperationException($"Expected a [{rowIndices.Length}, cols] tensor, got [{string.Join(",", Shape)}].");
        }

        int cols = Shape[1];
        var result = Zeros([targetRowCount, cols]);
        for (int r = 0; r < rowIndices.Length; r++)
        {
            int dstRow = rowIndices[r];
            for (int c = 0; c < cols; c++)
            {
                result._buffer[dstRow * cols + c] += _buffer[r * Strides[0] + c * Strides[1]];
            }
        }
        return result;
    }
}
