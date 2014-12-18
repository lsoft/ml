using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public class MetricsAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IMetrics _metrics;
        private readonly IDataSet _validationData;

        private readonly AccuracyCalculatorBatchIterator _batchIterator;

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

            _batchIterator = new AccuracyCalculatorBatchIterator();
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IDrawer drawer,
            out IAccuracyRecord accuracyRecord
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            //drawer allowed to be null

            if (drawer != null)
            {
                drawer.SetSize(
                    _validationData.Count
                    );
            }

            var totalErrorAcc = new KahanAlgorithm.Accumulator();

            _batchIterator.IterateByBatch(
                _validationData,
                forwardPropagation,
                (netResult, testItem) =>
                {
                    #region рисуем итем

                    if (drawer != null)
                    {
                        drawer.DrawItem(netResult);
                    }

                    #endregion

                    #region суммируем ошибку

                    var err = _metrics.Calculate(
                        testItem.Output,
                        netResult.NState
                        );

                    KahanAlgorithm.AddElement(ref totalErrorAcc, err);

                    #endregion
                });

            var totalError = totalErrorAcc.Sum;

            var perItemError = totalError / _validationData.Count;

            accuracyRecord = new MetricAccuracyRecord(
                epocheNumber ?? 0,
                perItemError);

            if (drawer != null)
            {
                drawer.Save();
            }

        }

    }
}