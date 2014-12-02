using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.Data.DataLoader;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
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

        public FeatureAndMetricsAccuracyCalculator(
            string mlpName,
            CLProvider clProvider,
            IMetrics metrics,
            IDataSet validationData,
            IDataItemFactory dataItemFactory
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

            _mlpName = mlpName;
            _clProvider = clProvider;
            _metrics = metrics;
            _validationData = validationData;
            _dataItemFactory = dataItemFactory;
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
                new MNISTVisualizer(),
                string.Format("{1}/_{0}_feature.bmp", epocheNumber != null ? epocheNumber.Value : -1, _mlpName),
                10,
                2f,
                900,
                false,
                true);

            netResults = forwardPropagation.ComputeOutput(_validationData);

            //����������� � ���, ����� � DataItem.Input - ���������� ����� (��������� �����),
            //� � DataItem.Output - �������� �����, � �� ������� - ������ ��������
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