using ILGPU.Runtime;
using Xunit;

namespace Tensor.Tests;

[Collection("GpuContext")]
public class GpuContextTests
{
    // TASK-031: ValidateAccelerator is a pure function of AcceleratorType
    // values specifically so these tests are deterministic regardless of
    // what GPU hardware (if any) is present on the machine running them -
    // the documented "no GPU found" error path must be provably correct
    // everywhere, not just verified by hand on one machine with a real GPU.

    [Fact]
    public void ValidateAccelerator_CpuSelectedWithoutFallbackAllowed_ThrowsNamingAvailableAccelerators()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            GpuContext.ValidateAccelerator(AcceleratorType.CPU, allowCpuFallback: false, [AcceleratorType.CPU]));

        Assert.Contains("No GPU accelerator", ex.Message);
        Assert.Contains("CPU", ex.Message);
    }

    [Fact]
    public void ValidateAccelerator_CpuSelectedWithFallbackAllowed_DoesNotThrow()
    {
        GpuContext.ValidateAccelerator(AcceleratorType.CPU, allowCpuFallback: true, [AcceleratorType.CPU]);
    }

    [Theory]
    [InlineData(AcceleratorType.Cuda)]
    [InlineData(AcceleratorType.OpenCL)]
    public void ValidateAccelerator_RealGpuSelected_DoesNotThrowRegardlessOfFallbackFlag(AcceleratorType gpuType)
    {
        GpuContext.ValidateAccelerator(gpuType, allowCpuFallback: false, [AcceleratorType.CPU, gpuType]);
        GpuContext.ValidateAccelerator(gpuType, allowCpuFallback: true, [AcceleratorType.CPU, gpuType]);
    }

    [Fact]
    public void ValidateAccelerator_ErrorMessage_DoesNotRepeatDuplicateAvailableAcceleratorTypes()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            GpuContext.ValidateAccelerator(AcceleratorType.CPU, allowCpuFallback: false, [AcceleratorType.CPU, AcceleratorType.CPU]));

        // Distinct(): a duplicate CPU device shouldn't produce "CPU, CPU" in the listed accelerators.
        Assert.DoesNotContain("CPU, CPU", ex.Message);
    }

    [Fact]
    public void GetAccelerator_WithCpuFallbackAllowed_ReturnsAWorkingAccelerator()
    {
        // The one hardware-dependent test here - but allowCpuFallback:
        // true means it succeeds on any machine, real GPU or not, proving
        // the plumbing (Context creation, device selection, accelerator
        // creation) genuinely works end to end rather than only compiling.
        try
        {
            var accelerator = GpuContext.GetAccelerator(allowCpuFallback: true);

            Assert.NotNull(accelerator);
            using var buffer = accelerator.Allocate1D<float>(4);
            buffer.CopyFromCPU([1f, 2f, 3f, 4f]);
            var result = buffer.GetAsArray1D();

            Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, result);
        }
        finally
        {
            GpuContext.Shutdown();
        }
    }

    [Fact]
    public void GetAccelerator_CalledTwice_ReturnsTheSameCachedAccelerator()
    {
        try
        {
            var first = GpuContext.GetAccelerator(allowCpuFallback: true);
            var second = GpuContext.GetAccelerator(allowCpuFallback: true);

            Assert.Same(first, second);
        }
        finally
        {
            GpuContext.Shutdown();
        }
    }

    [Fact]
    public void Shutdown_AllowsANewAcceleratorToBeCreatedAfterwards()
    {
        var first = GpuContext.GetAccelerator(allowCpuFallback: true);
        GpuContext.Shutdown();
        var second = GpuContext.GetAccelerator(allowCpuFallback: true);

        try
        {
            Assert.NotSame(first, second);
        }
        finally
        {
            GpuContext.Shutdown();
        }
    }
}
