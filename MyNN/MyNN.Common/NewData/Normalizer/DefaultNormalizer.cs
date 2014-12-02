using System;
using System.Linq;

namespace MyNN.Common.NewData.Normalizer
{
    [Serializable]
    public class DefaultNormalizer : INormalizer
    {
        /// <summary>
        /// Линейная нормализация [0;1]
        /// </summary>
        public void Normalize(
            float[] dataToNormalize,
            float bias = 0f
            )
        {
            if (dataToNormalize == null)
            {
                throw new ArgumentNullException("dataToNormalize");
            }
            var min = dataToNormalize.Min();
            var max = dataToNormalize.Max();

            for (var dd = 0; dd < dataToNormalize.Length; dd++)
            {
                var i = dataToNormalize[dd];
                i -= min;
                i /= (-min + max);
                dataToNormalize[dd] = i - bias;
            }
        }

        /// <summary>
        /// Гауссова нормализация
        /// mean = 0, variance = 1, standard deviation = 1
        /// </summary>
        public void GNormalize(
            float[] dataToNormalize
            )
        {
            if (dataToNormalize == null)
            {
                throw new ArgumentNullException("dataToNormalize");
            }

            var mean0 =
                (float) MathNet.Numerics.Statistics.Statistics.Mean(dataToNormalize.ToList().ConvertAll(j => (double) j));

            var variance0 =
                (float) MathNet.Numerics.Statistics.Statistics.Variance(dataToNormalize.ToList().ConvertAll(j => (double) j));

            var standardDeviation0 =
                (float) MathNet.Numerics.Statistics.Statistics.StandardDeviation(dataToNormalize.ToList().ConvertAll(j => (double) j));

            var sqrtVariance = (float) Math.Sqrt(variance0);

            //приводим к среднему = 0 и дисперсии = 1
            for (var i = 0; i < dataToNormalize.Length; i++)
            {
                dataToNormalize[i] -= mean0;
                dataToNormalize[i] /= sqrtVariance;
            }
        }
    }
}