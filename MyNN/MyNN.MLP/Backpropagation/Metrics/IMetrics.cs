using MyNN.Common.OpenCLHelper;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    public interface IMetrics
    {
        float Calculate(
            float[] desiredValues,
            float[] predictedValues
            );

        /// <summary>
        /// Calculate value of partial derivative by v2[v2Index]
        /// </summary>
        float CalculatePartialDerivativeByV2Index(
            float[] desiredValues,
            float[] predictedValues,
            int v2Index
            );

        /// <summary>
        /// Get kernel to evaluate partial derivative by v2[v2Index]
        /// </summary>
        string GetOpenCLPartialDerivative(
            string methodName,
            VectorizationSizeEnum vse,
            MemModifierEnum mme,
            int length
            );
    }
}
