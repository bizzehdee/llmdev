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

    /// <summary>
    /// TASK-031/032: an opt-in GPU-accelerated path for matmul, backed by
    /// ILGPU (see <see cref="GpuContext"/>) - CUDA, OpenCL, or ILGPU's own
    /// CPU accelerator, whichever <see cref="GpuContext"/> resolves to.
    /// Unlike <see cref="Optimised"/>, selecting this does *not* silently
    /// fall back to <see cref="Scalar"/> when no compatible GPU accelerator
    /// is available - <see cref="GpuContext.GetAccelerator"/> throws a
    /// clear, actionable error instead, since a caller selecting this
    /// backend is explicitly trying to demonstrate GPU execution, not
    /// asking for "whatever's fastest." Like <see cref="Optimised"/>,
    /// falls back to <see cref="Scalar"/> per-operand only for a genuinely
    /// different reason - a disk-backed tensor, the same case
    /// <see cref="Optimised"/> already declines.
    /// </summary>
    Gpu,
}
