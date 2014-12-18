using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Visualizer.Factory;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp
{
    public class FeatureAndMetricsAccuracyCalculator : IAccuracyCalculator
    {
        private readonly string _mlpName;
        private readonly CLProvider _clProvider;
        private readonly IMetrics _metrics;
        private readonly IDataSet _validationData;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IVisualizerFactory _visualizerFactory;

        public FeatureAndMetricsAccuracyCalculator(
            string mlpName,
            CLProvider clProvider,
            IMetrics metrics,
            IDataSet validationData,
            IDataItemFactory dataItemFactory,
            IVisualizerFactory visualizerFactory
            )
        {
            if (mlpName == null)
            {
                throw new ArgumentNullException("mlpName");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (metrics == null)
            {
                throw new ArgumentNullException("metrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (visualizerFactory == null)
            {
                throw new ArgumentNullException("visualizerFactory");
            }

            _mlpName = mlpName;
            _clProvider = clProvider;
            _metrics = metrics;
            _validationData = validationData;
            _dataItemFactory = dataItemFactory;
            _visualizerFactory = visualizerFactory;
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

            //в этом классе drawer не используетс€, так как фичи визуализируютс€ по другому

            var fff = new FileSystemFeatureVisualization(
                new NoRandomRandomizer(),
                new SerializationHelper().DeepClone(forwardPropagation.MLP),
                new ForwardPropagationFactory(
                    new CPUPropagatorComponentConstructor(
                        _clProvider,
                        VectorizationSizeEnum.VectorizationMode16)),
                _dataItemFactory
                );

            fff.Visualize(
                _visualizerFactory,
                string.Format("{1}/_{0}_feature.bmp", epocheNumber != null ? epocheNumber.Value : -1, _mlpName),
                10,
                2f,
                900,
                false,
                true);

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            //преобразуем в вид, когда в DataItem.Input - правильный ¬џ’ќƒ (обучаемый выход),
            //а в DataItem.Output - –≈јЋ№Ќџ… выход, а их разница - ошибка обучени€
            var d = new List<IDataItem>(_validationData.Count + 1);
            foreach (var pair in netResults.ZipEqualLength(_validationData))
            {
                var netResult = pair.Value1;
                var testItem = pair.Value2;

                d.Add(
                    _dataItemFactory.CreateDataItem(
                        testItem.Output,
                        netResult.NState));
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