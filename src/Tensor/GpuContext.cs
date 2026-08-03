using ILGPU;
using ILGPU.Runtime;

namespace Tensor;

/// <summary>
/// TASK-031: process-wide ILGPU context/accelerator management for
/// <see cref="TensorBackend.Gpu"/> - the plumbing a real GPU kernel
/// (TASK-032) builds on. ILGPU's <see cref="Context"/> and
/// <see cref="Accelerator"/> are genuinely process-lifetime, disposable
/// resources (unlike <see cref="TensorBackend.Optimised"/>'s
/// <c>System.Numerics.Tensors</c> path, which needs no such thing) - both
/// are created lazily on first use and torn down together via
/// <see cref="Shutdown"/>, not per-call, so repeated GPU ops in the same
/// process reuse one accelerator instead of paying context/accelerator
/// setup cost every time.
/// </summary>
public static class GpuContext
{
    private static readonly object Lock = new();
    private static Context? _context;
    private static Accelerator? _accelerator;

    /// <summary>
    /// Resolves (creating on first call) the accelerator this process's
    /// GPU backend uses. <paramref name="allowCpuFallback"/> controls what
    /// happens when no genuine GPU (CUDA or OpenCL) accelerator is found:
    /// <c>true</c> silently accepts ILGPU's own CPU accelerator (useful
    /// for tests/CI, which can't assume real GPU hardware is present);
    /// <c>false</c> - the default a CLI selecting <see cref="TensorBackend.Gpu"/>
    /// should use - throws <see cref="InvalidOperationException"/> naming
    /// every accelerator ILGPU actually found, rather than silently
    /// training on the CPU while claiming to demonstrate GPU execution.
    /// </summary>
    public static Accelerator GetAccelerator(bool allowCpuFallback = false)
    {
        lock (Lock)
        {
            if (_accelerator is not null)
            {
                return _accelerator;
            }

            _context ??= Context.Create(builder => builder.Default());
            var preferred = _context.GetPreferredDevice(preferCPU: allowCpuFallback);

            ValidateAccelerator(preferred.AcceleratorType, allowCpuFallback, _context.Devices.Select(d => d.AcceleratorType));

            _accelerator = preferred.CreateAccelerator(_context);
            return _accelerator;
        }
    }

    /// <summary>
    /// The decision logic <see cref="GetAccelerator"/> applies, extracted
    /// as a pure function of <see cref="AcceleratorType"/> values so it's
    /// testable without depending on what GPU hardware (if any) is
    /// actually present on the machine running the tests - the whole
    /// point being that this exact error path is provably correct on
    /// every machine, not just verified by hand on one with a real GPU.
    /// </summary>
    public static void ValidateAccelerator(AcceleratorType selected, bool allowCpuFallback, IEnumerable<AcceleratorType> available)
    {
        if (allowCpuFallback || selected != AcceleratorType.CPU)
        {
            return;
        }

        string availableList = string.Join(", ", available.Distinct());
        throw new InvalidOperationException(
            $"No GPU accelerator (CUDA or OpenCL) was found - only: {availableList}. " +
            "Pass an explicit CPU-fallback option to run ILGPU's CPU accelerator instead " +
            "(useful for testing the mechanism without real GPU hardware), or use --optimised for the CPU-only fast path.");
    }

    /// <summary>Disposes the accelerator and context, if created, and clears them so the next <see cref="GetAccelerator"/> call creates fresh ones. Mainly for test isolation.</summary>
    public static void Shutdown()
    {
        lock (Lock)
        {
            _accelerator?.Dispose();
            _accelerator = null;
            _context?.Dispose();
            _context = null;
        }
    }
}
