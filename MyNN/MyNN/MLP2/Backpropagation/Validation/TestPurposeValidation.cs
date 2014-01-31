using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.ForwardPropagation;

namespace MyNN.MLP2.Backpropagation.Validation
{
    public class TestPurposeValidation : IValidation
    {
        private readonly DataSet _validationData;

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

        public bool IsAuencoderDataSet
        {
            get
            {
                return
                    _validationData.IsAuencoderDataSet;
            }
        }

        public TestPurposeValidation(
            DataSet validationData)
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _validationData = validationData;
        }

        public float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<DataItem>(_validationData.Count + 1);
            for (int i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DataItem(
                        _validationData[i].Output,
                        netResults[i].State));
            }

            var metrics = new TestPurposeMetric();

            this.TotalError = d.Sum(j => metrics.Calculate(j.Input, j.Output));


            ulong resultsTotalSum = 0L;
            foreach (var nr in netResults)
            {
                foreach (var ni in nr.State)
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
                this.TotalError;
        }

    }

}
