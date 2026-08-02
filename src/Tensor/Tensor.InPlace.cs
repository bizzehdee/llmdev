namespace Tensor;

public sealed partial class Tensor
{
    /// <summary>
    /// Mutates this tensor's buffer directly: <c>this -= delta</c>,
    /// element-wise. Every other op in this file (and its siblings)
    /// deliberately returns a new tensor rather than mutating - this is the
    /// one exception, needed because a trained weight (a Variable's Value)
    /// can't simply be reassigned to a new Tensor object once optimizer
    /// state (in the training loop, TASK-012) is holding onto that same
    /// Variable by reference. Not part of the autodiff graph - only ever
    /// meant for an optimizer applying a computed update to a parameter,
    /// never for anything a Backward() pass needs to see.
    /// </summary>
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
}
