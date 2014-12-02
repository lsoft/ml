using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class RMSE : IMetrics
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
                d += (desiredValues[i] - predictedValues[i]) * (desiredValues[i] - predictedValues[i]);
            }

            return
                (float) Math.Sqrt(d/desiredValues.Length);
        }

        //производная взята отсюда http://www.wolframalpha.com/input/?i=d%28sqrt%281%2Fn*%28%28t1+-+y1%29%5E2+%2B+%28t2+-+y2%29%5E2+%2B+%28t3+-+y3%29%5E2%29%29%29%2Fd%28y2%29&a=*C.d-_*DerivativesWord-
        //только со знаком минус
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

            var acc = new KahanAlgorithm.Accumulator();

            for (var cc = 0; cc < desiredValues.Length; cc++)
            {
                var diff = desiredValues[cc] - predictedValues[cc];
                var sqDiff = diff*diff;
                KahanAlgorithm.AddElement(
                    ref acc,
                    sqDiff
                    );
            }

            var rSum = Math.Sqrt(acc.Sum);
            var pDiff = predictedValues[v2Index] - desiredValues[v2Index];

            var result = pDiff / rSum;

            return (float)result;
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
inline float{v} {METHOD_NAME}({MODIFIER} float{v}* v1, {MODIFIER} float{v}* v2, int v2Index)
{
    KahanAccumulator{v} acc = GetEmptyKahanAcc{v}();

    for (int cc = 0; cc < {LENGTH}; cc++)
    {
        float{v} diff = v1[cc] - v2[cc];

        float{v} sqDiff = diff * diff;

        KahanAddElement{v}(
            &acc,
            sqDiff
            );
    }

    float{v} rSum = sqrt(acc.Sum);
    float{v} pDiff = v2[v2Index] - v1[v2Index];

    float{v} result = pDiff / rSum;

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

            result = result.Replace(
                "{LENGTH}",
                length.ToString()
                );

            return result;
        }
    }
}
