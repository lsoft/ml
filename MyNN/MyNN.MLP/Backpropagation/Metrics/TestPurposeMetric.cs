using System;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class TestPurposeMetric : IMetrics
    {
        public float Calculate(float[] desiredValues, float[] predictedValues)
        {
            if (desiredValues.Length != predictedValues.Length)
            {
                throw new InvalidOperationException("v1.Length != v2.Length");
            }

            var d = 0.0f;
            for (var i = 0; i < desiredValues.Length; i++)
            {
                d += (desiredValues[i] - predictedValues[i]);
            }

            return
                d;
        }

        public float CalculatePartialDerivativeByV2Index(float[] desiredValues, float[] predictedValues, int v2Index)
        {
            return predictedValues[v2Index] - desiredValues[v2Index];
        }

        public string GetOpenCLPartialDerivative(string methodName, VectorizationSizeEnum vse, MemModifierEnum mme, int length)
        {
            throw new NotImplementedException();
        }
    }
}
