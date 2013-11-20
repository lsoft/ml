using System;

namespace MyNN.NeuralNet.Train.Metrics
{
    [Serializable]
    public class HalfSquaredEuclidianDistance : IMetrics
    {
        public float Calculate(float[] v1, float[] v2)
        {
            var d = 0.0f;
            for (int i = 0; i < v1.Length; i++)
            {
                d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            }
            return 0.5f * d;
        }

        public float CalculatePartialDerivaitveByV2Index(float[] v1, float[] v2, int v2Index)
        {
            return v2[v2Index] - v1[v2Index];
        }
    }
}
