namespace Training;

public interface IOptimizer
{
    /// <summary>Applies one update to every tracked parameter, using its current gradient.</summary>
    void Step();

    /// <summary>Zeroes every tracked parameter's gradient - call between Step() and the next Backward().</summary>
    void ZeroGrad();
}
