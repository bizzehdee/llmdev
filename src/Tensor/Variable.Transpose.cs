namespace Tensor;

public sealed partial class Variable
{
    public Variable Transpose(int dim0, int dim1)
    {
        var value = Value.Transpose(dim0, dim1);
        return FromOp(value, result => () =>
        {
            // Transposing is its own inverse for the same pair of dims, so
            // the upstream gradient just gets transposed straight back.
            AccumulateGradient(result.Gradient.Transpose(dim0, dim1));
        }, this);
    }
}
