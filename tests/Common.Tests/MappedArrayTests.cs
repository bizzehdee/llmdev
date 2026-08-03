using Common;
using Xunit;

namespace Common.Tests;

public class MappedArrayTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "common-tests-scratch");

    static MappedArrayTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void Constructor_ExposesLength()
    {
        using var array = new MappedArray<int>(5, ScratchDirectory);

        Assert.Equal(5, array.Length);
    }

    [Fact]
    public void Indexer_GetSetRoundtrips()
    {
        using var array = new MappedArray<int>(4, ScratchDirectory);

        array[0] = 10;
        array[1] = -5;
        array[3] = int.MaxValue;

        Assert.Equal(10, array[0]);
        Assert.Equal(-5, array[1]);
        Assert.Equal(0, array[2]); // never written, should be zero-initialised
        Assert.Equal(int.MaxValue, array[3]);
    }

    [Fact]
    public void SupportsFloatElementsToo()
    {
        using var array = new MappedArray<float>(3, ScratchDirectory);

        array[0] = 1.5f;
        array[1] = -2.25f;

        Assert.Equal(1.5f, array[0]);
        Assert.Equal(-2.25f, array[1]);
    }

    [Fact]
    public void Dispose_DeletesTheBackingScratchFile()
    {
        string before = Directory.GetFiles(ScratchDirectory, "mapped-*.scratch").Length.ToString();
        var array = new MappedArray<int>(2, ScratchDirectory);
        Assert.NotEmpty(Directory.GetFiles(ScratchDirectory, "mapped-*.scratch"));

        array.Dispose();

        // FileOptions.DeleteOnClose means the count returns to what it was before.
        Assert.Equal(before, Directory.GetFiles(ScratchDirectory, "mapped-*.scratch").Length.ToString());
    }

    [Fact]
    public void Dispose_CalledTwice_IsASafeNoOp()
    {
        var array = new MappedArray<int>(2, ScratchDirectory);
        array[0] = 42;

        array.Dispose();
        var exception = Record.Exception(() => array.Dispose());

        Assert.Null(exception);
    }
}
