namespace Tensor;

/// <summary>
/// Which implementation <see cref="Tensor"/>'s hot ops use. See
/// <see cref="Tensor.Backend"/> for how this is selected, and PLAN.md
/// stage 9 / TASK-015 for the full rationale behind this being the one
/// narrow, opt-in exception to the project's no-libraries rule.
/// </summary>
public enum TensorBackend
{
    /// <summary>The hand-written scalar implementation. Always correct, always available, the default.</summary>
    Scalar = 0,

    /// <summary>
    /// An opt-in fast path for ops where it applies (matmul), backed by
    /// <c>System.Numerics.Tensors.TensorPrimitives</c>. Falls back to
    /// <see cref="Scalar"/> transparently wherever it doesn't apply (e.g.
    /// a disk-backed operand - see <see cref="IFloatBuffer.TryGetSpan"/>),
    /// so selecting this is always safe, just not always faster.
    /// </summary>
    Optimised,
}
