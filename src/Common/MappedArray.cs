using System.IO.MemoryMappedFiles;

namespace Common;

/// <summary>
/// A fixed-length array of an unmanaged value type, backed by a
/// memory-mapped scratch file instead of the managed heap. The OS treats
/// its pages as reclaimable file-backed memory (dropped or written back on
/// reclaim) rather than anonymous process memory (which can only be
/// reclaimed via swap) - so a large array here doesn't push the machine
/// towards OOM the way a plain array of the same size would under memory
/// pressure.
///
/// This only holds if <paramref name="scratchDirectory"/> in the
/// constructor is on real disk. A directory backed by <c>tmpfs</c> (e.g.
/// <c>/tmp</c> on many Linux distros, including this dev machine) is itself
/// RAM, so a "scratch file" there is no different from a plain heap array -
/// the caller is responsible for pointing this at genuine disk.
/// </summary>
public sealed unsafe class MappedArray<T> : IDisposable where T : unmanaged
{
    private readonly string _path;
    private readonly FileStream _file;
    private readonly MemoryMappedFile _mappedFile;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _pointer;
    private bool _disposed;

    public int Length { get; }

    public MappedArray(int length, string scratchDirectory)
    {
        Length = length;
        long byteLength = (long)length * sizeof(T);

        _path = Path.Combine(scratchDirectory, $"mapped-{Guid.NewGuid():N}.scratch");
        _file = new FileStream(_path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, bufferSize: 1, FileOptions.DeleteOnClose);
        _file.SetLength(Math.Max(byteLength, 1));

        _mappedFile = MemoryMappedFile.CreateFromFile(_file, mapName: null, byteLength, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: true);
        _view = _mappedFile.CreateViewAccessor(0, byteLength, MemoryMappedFileAccess.ReadWrite);

        byte* ptr = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
        _pointer = ptr;
    }

    public T this[int index]
    {
        get => ((T*)_pointer)[index];
        set => ((T*)_pointer)[index] = value;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;

        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _mappedFile.Dispose();
        _file.Dispose(); // FileOptions.DeleteOnClose removes the backing temp file here
    }
}
