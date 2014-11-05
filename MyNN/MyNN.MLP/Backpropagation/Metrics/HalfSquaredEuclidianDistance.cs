using System;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class HalfSquaredEuclidianDistance : IMetrics
    {
        public float Calculate(
            float[] v1,
            float[] v2
            )
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
                throw new InvalidOperationException("v1.Length != v2.Length");
            }

            var d = 0.0f;
            for (var i = 0; i < v1.Length; i++)
            {
                var diff = v1[i] - v2[i];
                d += diff * diff;
            }

            return 0.5f * d;
        }

        public float CalculatePartialDerivativeByV2Index(
            float[] v1,
            float[] v2,
            int v2Index
            )
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
                throw new InvalidOperationException("v1.Length != v2.Length");
            }
            if (v2Index >= v2.Length)
            {
                throw new ArgumentException("v2Index >= v2.Length");
            }

            return
                v2[v2Index] - v1[v2Index];
        }

        public string GetOpenCLPartialDerivative(
            string methodName,
            VectorizationSizeEnum vse,
            int length
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv* v1, floatv* v2, int v2Index)
{
    floatv result = v2[v2Index] - v1[v2Index];

    return result;
}
";

            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }
}
