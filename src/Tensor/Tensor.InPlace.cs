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

    /// <summary>
    /// TASK-036: moves this tensor's storage onto the GPU in place - the
    /// same reason <see cref="SubtractInPlace"/>/<see cref="LoadInPlace"/>
    /// mutate rather than return a new <see cref="Tensor"/>: a model
    /// parameter (a <c>Variable</c>'s <c>Value</c>) can't simply be
    /// reassigned once an optimizer or checkpoint loader already holds
    /// that same <c>Variable</c> by reference. A no-op if already
    /// device-resident. Disposes the old (heap or disk-backed) buffer once
    /// its data has been copied, so this doesn't leak the tensor's
    /// previous storage.
    /// </summary>
    public void MoveToGpuInPlace()
    {
        if (_buffer is GpuFloatBuffer)
        {
            return;
        }

        var gpuBuffer = new GpuFloatBuffer(Length);
        if (_buffer.TryGetSpan(out var hostSpan))
        {
            gpuBuffer.CopyFromHost(hostSpan);
        }
        else
        {
            for (int i = 0; i < Length; i++)
            {
                gpuBuffer[i] = _buffer[i];
            }
        }

        _buffer.Dispose();
        _buffer = gpuBuffer;
    }

    /// <summary>The inverse of <see cref="MoveToGpuInPlace"/> - moves this tensor's storage back onto the heap in place. A no-op if not currently device-resident.</summary>
    public void MoveToHostInPlace()
    {
        if (_buffer is not GpuFloatBuffer gpuBuffer)
        {
            return;
        }

        var heapBuffer = new HeapFloatBuffer(Length);
        heapBuffer.TryGetSpan(out var hostSpan); // HeapFloatBuffer always supports this.
        gpuBuffer.CopyToHost(hostSpan);

        _buffer.Dispose();
        _buffer = heapBuffer;
    }
}
