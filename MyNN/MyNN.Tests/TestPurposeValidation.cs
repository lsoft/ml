using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.Tests
{
    public class TestPurposeValidation : IValidation
    {
        private readonly IDataSet _validationData;
        private readonly DataItemFactory _dataItemFactory;

        public float TotalError
        {
            get;
            private set;
        }

        public ulong TotalSum
        {
            get;
            private set;
        }

        public TestPurposeValidation(
            IDataSet validationData)
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _validationData = validationData;
            _dataItemFactory = new DataItemFactory();
        }

        public IAccuracyRecord Validate(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IArtifactContainer epocheContainer
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            var netResults = forwardPropagation.ComputeOutput(_validationData);


            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<IDataItem>(_validationData.Count + 1);
            foreach (var pair in netResults.ZipEqualLength(_validationData))
            {
                var netResult = pair.Value1;
                var testItem = pair.Value2;

                d.Add(
                    _dataItemFactory.CreateDataItem(
                        testItem.Output,
                        netResult.NState
                        ));
            }


            var metrics = new TestPurposeMetric();

            this.TotalError = d.Sum(j => metrics.Calculate(j.Input, j.Output));

            ulong resultsTotalSum = 0L;
            foreach (var nr in netResults)
            {
                foreach (var ni in nr.NState)
                {
                    var bytes = BitConverter.GetBytes(ni);
                    var uinteger = BitConverter.ToUInt32(bytes, 0);

                    resultsTotalSum += uinteger;
                }
            }

            ulong weightTotalSum = 0L;
            //foreach (var layer in forwardPropagation.MLP.Layers.Skip(1))
            //{
            //    foreach (var neuron in layer.Neurons)
            //    {
            //        foreach (var weight in neuron.Weights)
            //        {
            //            var bytes = BitConverter.GetBytes(weight);
            //            var uinteger = BitConverter.ToUInt32(bytes, 0);

            //            weightTotalSum += uinteger;
            //        }
            //    }
            //}

            this.TotalSum = resultsTotalSum + weightTotalSum;

            return
                null;
        }

    }

}
