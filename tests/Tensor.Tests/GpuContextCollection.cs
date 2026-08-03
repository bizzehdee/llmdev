using Xunit;

namespace Tensor.Tests;

/// <summary>
/// TASK-032: <see cref="Tensor.GpuContext"/> is a process-wide singleton
/// (one shared ILGPU <c>Context</c>/<c>Accelerator</c>), and
/// <see cref="GpuContextTests"/> deliberately calls
/// <see cref="Tensor.GpuContext.Shutdown"/> to test it - which would race
/// with any other test concurrently running a GPU-backed matmul on that
/// same shared accelerator, since xUnit runs different test classes in
/// parallel by default. Every class that touches
/// <see cref="TensorBackend.Gpu"/> shares this one named collection so
/// xUnit runs them sequentially relative to each other (only these three
/// - unrelated test classes elsewhere are untouched and still run in
/// parallel).
/// </summary>
[CollectionDefinition("GpuContext")]
public class GpuContextCollection;
