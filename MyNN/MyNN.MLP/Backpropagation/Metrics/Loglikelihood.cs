using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class Loglikelihood : IMetrics
    {
        private const float Epsilon = 0.00001f;

        public float Calculate(float[] desiredValues, float[] predictedValues)
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
                throw new ArgumentException("desiredValues.Length != predictedValues.Length");
            }

            var dacc = new KahanAlgorithm.Accumulator();

            for (var i = 0; i < desiredValues.Length; i++)
            {
                var desiredValue = desiredValues[i];
                desiredValue = Math.Min(Math.Max(Epsilon, desiredValue), 1f - Epsilon);

                var predictedValue = predictedValues[i];
                predictedValue = Math.Min(Math.Max(Epsilon, predictedValue), 1f - Epsilon);

                var delta = (float) (desiredValue*Math.Log(predictedValue) + (1 - desiredValue)*Math.Log(1 - predictedValue));

                KahanAlgorithm.AddElement(
                    ref dacc,
                    delta
                    );
            }

            var d = dacc.Sum;

            d /= desiredValues.Length;

            return -d;
        }

        public float CalculatePartialDerivativeByV2Index(
            float[] desiredValues, 
            float[] predictedValues, 
            int v2Index)
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
                throw new ArgumentException("desiredValues.Length != predictedValues.Length");
            }
            if (v2Index >= predictedValues.Length)
            {
                throw new ArgumentException("v2Index >= predictedValues.Length");
            }

            var desiredValue = desiredValues[v2Index];
            desiredValue = Math.Min(Math.Max(Epsilon, desiredValue), 1f - Epsilon);

            var predictedValue = predictedValues[v2Index];
            predictedValue = Math.Min(Math.Max(Epsilon, predictedValue), 1f - Epsilon);

            return
                -(desiredValue / predictedValue - (1 - desiredValue) / (1 - predictedValue));
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
    const float{v} Epsilon = 0.00001;
    const float{v} Epsilonm1 = 1 - 0.00001;

    float{v} desiredValue = desiredValues[v2Index];
    desiredValue = min(max(Epsilon, desiredValue), Epsilonm1);

    float{v} predictedValue = predictedValues[v2Index];
    predictedValue = min(max(Epsilon, predictedValue), Epsilonm1);

    float{v} result = -(desiredValue / predictedValue - (1 - desiredValue) / (1 - predictedValue));

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
