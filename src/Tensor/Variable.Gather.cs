namespace Tensor;

public sealed partial class Variable
{
    /// <summary>Differentiable row lookup - see Tensor.GatherRows/ScatterAddRows.</summary>
    public Variable GatherRows(int[] rowIndices)
    {
        var value = Value.GatherRows(rowIndices);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.ScatterAddRows(rowIndices, Value.Shape[0]));
        }, this);
    }

    /// <summary>Differentiable per-row column pick - see Tensor.GatherColumns/ScatterAddColumns.</summary>
    public Variable GatherColumns(int[] columnIndices)
    {
        var value = Value.GatherColumns(columnIndices);
        return FromOp(value, result => () =>
        {
            AccumulateGradient(result.Gradient.ScatterAddColumns(columnIndices, Value.Shape[1]));
        }, this);
    }
}
