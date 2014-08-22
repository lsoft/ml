using System;

namespace MyNN.MLP2.Backpropagation.Metrics
{
    [Serializable]
    public class Loglikelihood : IMetrics
    {
        public float Calculate(float[] v1, float[] v2)
        {
            if (v1 == null)
            {
                throw new ArgumentNullException("v1");
            }
            if (v2 == null)
            {
                throw new ArgumentNullException("v2");
            }
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("v1.Length != v2.Length");
            }

            //!!! не усредняется значение по длине векторов!
            //!!! нет нумерикал стабилити: https://www.kaggle.com/wiki/LogarithmicLoss
            //!!! использовать Kahan

            var d = 0.0f;
            for (var i = 0; i < v1.Length; i++)
            {
                d += (float)(v1[i] * Math.Log(v2[i]) + (1 - v1[i]) * Math.Log(1 - v2[i]));
            }
            return -d;
        }

        public float CalculatePartialDerivativeByV2Index(
            float[] v1, 
            float[] v2, 
            int v2Index)
        {
            if (v1 == null)
            {
                throw new ArgumentNullException("v1");
            }
            if (v2 == null)
            {
                throw new ArgumentNullException("v2");
            }
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("v1.Length != v2.Length");
            }
            if (v2Index >= v2.Length)
            {
                throw new ArgumentException("v2Index >= v2.Length");
            }

            //!!! нет нумерикал стабилити!

            return 
                -(v1[v2Index] / v2[v2Index] - (1 - v1[v2Index]) / (1 - v2[v2Index]));
        }
    }
}
