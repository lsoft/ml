using System;
using MyNN.Common.OpenCLHelper;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class HalfSquaredEuclidianDistance : IMetrics
    {
        public float Calculate(
            float[] desiredValues,
            float[] predictedValues
            )
        {
            if (desiredValues == null)
            {
                throw new ArgumentNullException("desiredValues");
            }
            if (predictedValues == null)
            {
                throw new ArgumentNullException("predictedValues");
            }
            if (desiredValues.Length != predictedValues.Length)
            {
                throw new InvalidOperationException("v1.Length != v2.Length");
            }

            var d = 0.0f;
            for (var i = 0; i < desiredValues.Length; i++)
            {
                var diff = desiredValues[i] - predictedValues[i];
                d += diff * diff;
            }

            return 0.5f * d;
        }

        public float CalculatePartialDerivativeByV2Index(
            float[] desiredValues,
            float[] predictedValues,
            int v2Index
            )
        {
            if (desiredValues == null)
            {
                throw new ArgumentNullException("desiredValues");
            }
            if (predictedValues == null)
            {
                throw new ArgumentNullException("predictedValues");
            }
            if (desiredValues.Length != predictedValues.Length)
            {
                throw new InvalidOperationException("v1.Length != v2.Length");
            }
            if (v2Index >= predictedValues.Length)
            {
                throw new ArgumentException("v2Index >= v2.Length");
            }

            return
                predictedValues[v2Index] - desiredValues[v2Index];
        }

        public string GetOpenCLPartialDerivative(
            string methodName,
            VectorizationSizeEnum vse,
            MemModifierEnum mme,
            int length
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline float{v} {METHOD_NAME}({MODIFIER} float{v}* desiredValues, {MODIFIER} float{v}* predictedValues, int v2Index)
{
    float{v} result = predictedValues[v2Index] - desiredValues[v2Index];

    return result;
}
";

            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);
            var mm = MemModifierHelper.GetModifierSuffix(mme);

            var result = methodBody;

            result = result.Replace(
                "{v}",
                string.Format(
                    "{0}",
                    vsize));

            result = result.Replace(
                "{MODIFIER}",
                mm
                );

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }
}
