using System;

namespace MyNN.NeuralNet.Train.Metrics
{
    [Serializable]
    public class Loglikelihood : IMetrics
    {
        public float Calculate(float[] v1, float[] v2)
        {
            var d = 0.0f;
            for (int i = 0; i < v1.Length; i++)
            {
                d += (float)(v1[i] * Math.Log(v2[i]) + (1 - v1[i]) * Math.Log(1 - v2[i]));
            }
            return -d;
        }

        public float CalculatePartialDerivaitveByV2Index(float[] v1, float[] v2, int v2Index)
        {
            return -(v1[v2Index] / v2[v2Index] - (1 - v1[v2Index]) / (1 - v2[v2Index]));
        }
    }
}
