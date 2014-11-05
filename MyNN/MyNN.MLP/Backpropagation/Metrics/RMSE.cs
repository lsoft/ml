using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class RMSE : IMetrics
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
                d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            }

            return
                (float) Math.Sqrt(d/v1.Length);
        }

        //производная взята отсюда http://www.wolframalpha.com/input/?i=d%28sqrt%281%2Fn*%28%28t1+-+y1%29%5E2+%2B+%28t2+-+y2%29%5E2+%2B+%28t3+-+y3%29%5E2%29%29%29%2Fd%28y2%29&a=*C.d-_*DerivativesWord-
        //только со знаком минус
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

            var acc = new KahanAlgorithm.Accumulator();

            for (var cc = 0; cc < v1.Length; cc++)
            {
                var diff = v1[cc] - v2[cc];
                var sqDiff = diff*diff;
                KahanAlgorithm.AddElement(
                    ref acc,
                    sqDiff
                    );
            }

            var rSum = Math.Sqrt(acc.Sum);
            var rN = Math.Sqrt(v1.Length);
            var pDiff = v2[v2Index] - v1[v2Index];

            var result = pDiff/rN/rSum;

            return (float)result;
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
    KahanAccumulatorv acc = GetEmptyKahanAccv();

    for (int cc = 0; cc < {LENGTH}; cc++)
    {
        floatv diff = v1[cc] - v2[cc];

        floatv sqDiff = diff * diff;

        KahanAddElementv(
            &acc,
            sqDiff
            );
    }

    floatv rSum = sqrt(acc.Sum);
    floatv rN = sqrt({LENGTH});
    floatv pDiff = v2[v2Index] - v1[v2Index];

    floatv result = pDiff / rN / rSum;

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
                "KahanAccumulatorv",
                string.Format(
                    "KahanAccumulator{0}",
                    vsize));

            result = result.Replace(
                "GetEmptyKahanAccv",
                string.Format(
                    "GetEmptyKahanAcc{0}",
                    vsize));

            result = result.Replace(
                "KahanAddElementv",
                string.Format(
                    "KahanAddElement{0}",
                    vsize));

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
