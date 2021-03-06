using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;

namespace MyNN.MLP.Backpropagation.Metrics
{
    [Serializable]
    public class MultiClassLogLoss : IMetrics
    {
        private readonly double _extremeValue;// = 10e-15d;
        private readonly double _extremeValueM1;// = 1 - 10e-15d;

        public MultiClassLogLoss(
            double extremeValue = 10e-15d
            )
        {
            _extremeValue = extremeValue;
            _extremeValueM1 = 1d - extremeValue;

            ConsoleAmbientContext.Console.WriteWarning("MultiClassLogLoss is not verified and completed in terms of first derivative");
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

            var sum = predictedValues.Sum();

            var acc = new KahanAlgorithm.Accumulator();

            for (var j = 0; j < desiredValues.Length; j++)
            {
                var normalizedprection = predictedValues[j]/sum;
                var p = Math.Max(Math.Min(normalizedprection, _extremeValueM1), _extremeValue);

                var atom = desiredValues[j] * Math.Log(p, Math.E);

                KahanAlgorithm.AddElement(ref acc, (float)atom);
            }

            return
                - acc.Sum;
        }

        public float CalculatePartialDerivativeByV2Index(float[] desiredValues, float[] predictedValues, int v2Index)
        {
            throw new NotSupportedException("� ������ ������� �������� �� ��������������");
        }

        public string GetOpenCLPartialDerivative(string methodName, VectorizationSizeEnum vse, MemModifierEnum mme, int length)
        {
            throw new NotSupportedException("� ������ ������� �������� �� ��������������");
        }
    }
}