using Xunit;

namespace Tensor.Tests;

[Collection("GpuContext")]
public class GpuFloatBufferTests
{
    [Fact]
    public void Indexer_RoundTripsASingleValue()
    {
        using var buffer = new GpuFloatBuffer(4);

        buffer[2] = 42f;

        Assert.Equal(42f, buffer[2]);
    }

    [Fact]
    public void Constructor_ZeroInitialisesLikeTheOtherBufferKinds()
    {
        using var buffer = new GpuFloatBuffer(8);

        for (int i = 0; i < buffer.Length; i++)
        {
            Assert.Equal(0f, buffer[i]);
        }
    }

    [Fact]
    public void CopyFromHost_ThenCopyToHost_RoundTripsExactly()
    {
        using var buffer = new GpuFloatBuffer(5);
        float[] values = [1f, 2f, 3f, 4f, 5f];

        buffer.CopyFromHost(values);
        var result = new float[5];
        buffer.CopyToHost(result);

        Assert.Equal(values, result);
    }

    [Fact]
    public void TryGetSpan_AlwaysDeclines()
    {
        using var buffer = new GpuFloatBuffer(4);

        bool succeeded = buffer.TryGetSpan(out var span);

        Assert.False(succeeded);
        Assert.Equal(0, span.Length);
    }

    [Fact]
    public void Indexer_SetThenGet_MultipleDistinctIndices()
    {
        using var buffer = new GpuFloatBuffer(3);

        buffer[0] = 10f;
        buffer[1] = 20f;
        buffer[2] = 30f;

        Assert.Equal(10f, buffer[0]);
        Assert.Equal(20f, buffer[1]);
        Assert.Equal(30f, buffer[2]);
    }
}
