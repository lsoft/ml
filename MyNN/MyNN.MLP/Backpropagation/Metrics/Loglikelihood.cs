using System;
using MyNN.Common.OpenCLHelper;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class Loglikelihood : IMetrics
    {
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

            //!!! не усредняется значение по длине векторов!
            //!!! нет нумерикал стабилити: https://www.kaggle.com/wiki/LogarithmicLoss
            //!!! использовать Kahan

            var d = 0.0f;
            for (var i = 0; i < desiredValues.Length; i++)
            {
                d += (float)(desiredValues[i] * Math.Log(predictedValues[i]) + (1 - desiredValues[i]) * Math.Log(1 - predictedValues[i]));
            }
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

            //!!! нет нумерикал стабилити!

            return 
                -(desiredValues[v2Index] / predictedValues[v2Index] - (1 - desiredValues[v2Index]) / (1 - predictedValues[v2Index]));
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

            //!!! нет нумерикал стабилити!

            const string methodBody = @"
inline float{v} {METHOD_NAME}({MODIFIER} float{v}* desiredValues, {MODIFIER} float{v}* predictedValues, int v2Index)
{
    float{v} result = -(desiredValues[v2Index] / predictedValues[v2Index] - (1 - desiredValues[v2Index]) / (1 - predictedValues[v2Index]));

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
