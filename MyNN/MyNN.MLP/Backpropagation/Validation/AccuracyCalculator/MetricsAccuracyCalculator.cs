using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
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

            //преобразуем в вид, когда в DenseDataItem.Input - правильный ¬џ’ќƒ (обучаемый выход),
            //а в DenseDataItem.Output - –≈јЋ№Ќџ… выход, а их разница - ошибка обучени€
            var d = new List<IDataItem>(_validationData.Count + 1);
            for (var i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DenseDataItem(
                        _validationData[i].Output,
                        netResults[i].NState));
            }

            var totalError = d.AsParallel().Sum(
                j => _metrics.Calculate(j.Input, j.Output));

            var perItemError = totalError / _validationData.Count;

            accuracyRecord = new MetricAccuracyRecord(
                epocheNumber ?? 0,
                perItemError);
        }

    }
}