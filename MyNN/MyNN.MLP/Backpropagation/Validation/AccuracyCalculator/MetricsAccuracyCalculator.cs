using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Other;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public class MetricsAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IMetrics _metrics;
        private readonly IDataSet _validationData;

        public MetricsAccuracyCalculator(
            IMetrics metrics,
            IDataSet validationData
            )
        {
            if (metrics == null)
            {
                throw new ArgumentNullException("metrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _metrics = metrics;
            _validationData = validationData;
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            out List<ILayerState> netResults,
            out IAccuracyRecord accuracyRecord
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            netResults = forwardPropagation.ComputeOutput(_validationData);

            //преобразуем в вид, когда в DataItem.Input - правильный ¬џ’ќƒ (обучаемый выход),
            //а в DataItem.Output - –≈јЋ№Ќџ… выход, а их разница - ошибка обучени€
            var d = new List<Pair<float[], float[]>>(_validationData.Count + 1);
            foreach (var pair in netResults.ZipEqualLength(_validationData))
            {
                var netResult = pair.Value1;
                var testItem = pair.Value2;

                d.Add(
                    new Pair<float[], float[]>(
                        testItem.Output,
                        netResult.NState));
            }

            var totalError = d.AsParallel().Sum(
                j => _metrics.Calculate(j.First, j.Second));

            var perItemError = totalError / _validationData.Count;

            accuracyRecord = new MetricAccuracyRecord(
                epocheNumber ?? 0,
                perItemError);
        }

    }
}