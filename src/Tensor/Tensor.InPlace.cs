namespace Tensor;

/// <summary>
/// Every op elsewhere in the tensor engine deliberately returns a new
/// tensor rather than mutating - the two methods here are the only
/// exceptions, both needed because a trained weight (a Variable's Value)
/// can't simply be reassigned to a new Tensor object once an optimizer or
/// checkpoint loader is holding that same Variable by reference. Neither
/// is part of the autodiff graph, and neither should ever be called on
/// anything a Backward() pass still needs to see.
/// </summary>
public sealed partial class Tensor
{
    /// <summary>this -= delta, element-wise. Used by an optimizer applying a computed update to a parameter.</summary>
    public void SubtractInPlace(Tensor delta)
    {
        if (!Shape.SequenceEqual(delta.Shape))
        {
            throw new InvalidOperationException($"Shape mismatch: [{string.Join(",", Shape)}] vs [{string.Join(",", delta.Shape)}].");
        }

        for (int i = 0; i < Length; i++)
        {
            _buffer[i] -= delta._buffer[i];
        }
    }

    /// <summary>Overwrites this tensor's values in place. Used to restore a parameter's value from a checkpoint.</summary>
    public void LoadInPlace(float[] values)
    {
        if (values.Length != Length)
        {
            throw new InvalidOperationException($"Expected {Length} values, got {values.Length}.");
        }

        for (int i = 0; i < Length; i++)
        {
            _buffer[i] = values[i];
        }
    }
}
