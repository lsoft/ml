using System;
using System.Linq;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class AUC : IMetrics
    {
        public AUC()
        {
            ConsoleAmbientContext.Console.WriteWarning("AUC is not verified and completed in terms of first derivative");
        }

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

            var lSorted = desiredValues
                .Zip(predictedValues, (actual, pred) => new { ActualValue = actual < 0.5f ? 0 : 1, PredictedValue = pred })
                .OrderBy(ap => ap.PredictedValue)
                .ToArray();

            long n = lSorted.Length;
            long ones = lSorted.Sum(v => v.ActualValue);

            if (0 == ones || n == ones)
            {
                return 1;
            }

            long tp_prev = ones;
            long tp = ones;
            long accum = 0;
            long tn = 0;
            double threshold = lSorted[0].PredictedValue;

            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(lSorted[i].PredictedValue - threshold) > float.Epsilon)
                {
                    // threshold changes
                    threshold = lSorted[i].PredictedValue;
                    accum += tn * (tp + tp_prev); //2 * the area of  trapezoid
                    tp_prev = tp;
                    tn = 0;
                }

                tn += 1 - lSorted[i].ActualValue; // x-distance between adjacent points
                tp -= lSorted[i].ActualValue;
            }

            accum += tn * (tp + tp_prev); // 2 * the area of trapezoid

            return
                accum / (float)(2 * ones * (n - ones));
        }

        public float CalculatePartialDerivativeByV2Index(float[] desiredValues, float[] predictedValues, int v2Index)
        {
            throw new NotSupportedException("В данной метрике обучение не поддерживается");
        }

        public string GetOpenCLPartialDerivative(string methodName, VectorizationSizeEnum vse, MemModifierEnum mme, int length)
        {
            throw new NotSupportedException("В данной метрике обучение не поддерживается");
        }
    }
}