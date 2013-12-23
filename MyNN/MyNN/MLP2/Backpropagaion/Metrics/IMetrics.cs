namespace MyNN.MLP2.Backpropagaion.Metrics
{
    public interface IMetrics
    {
        float Calculate(float[] v1, float[] v2);

        /// <summary>
        /// Calculate value of partial derivative by v2[v2Index]
        /// </summary>
        float CalculatePartialDerivaitveByV2Index(float[] v1, float[] v2, int v2Index);
    }
}
