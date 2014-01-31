using System;

namespace MyNN.MLP2.Backpropagation.Metrics
{
    [Serializable]
    public class TestPurposeMetric : IMetrics
    {
        public float Calculate(float[] v1, float[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new InvalidOperationException("v1.Length != v2.Length");
            }

            var d = 0.0f;
            for (var i = 0; i < v1.Length; i++)
            {
                d += (v1[i] - v2[i]);
            }

            return
                d;
        }

        public float CalculatePartialDerivaitveByV2Index(float[] v1, float[] v2, int v2Index)
        {
            return v2[v2Index] - v1[v2Index];
        }
    }
}
