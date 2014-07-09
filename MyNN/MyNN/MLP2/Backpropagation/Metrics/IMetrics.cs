namespace MyNN.MLP2.Backpropagation.Metrics
{
    public interface IMetrics
    {
        float Calculate(float[] v1, float[] v2);

        /// <summary>
        /// Calculate value of partial derivative by v2[v2Index]
        /// </summary>
        float CalculatePartialDerivativeByV2Index(float[] v1, float[] v2, int v2Index);
    }
}
