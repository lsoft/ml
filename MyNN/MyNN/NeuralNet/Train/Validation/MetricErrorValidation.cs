using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Train.Metrics;

namespace MyNN.NeuralNet.Train.Validation
{
    public class MetricErrorValidation : IValidation
    {
        private readonly IMetrics _errorMetrics;
        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;
        private readonly int _domainCountThreshold;
        private readonly int _outputLength;

        private float _bestPerItemError = float.MaxValue;
        public float BestPerItemError
        {
            get
            {
                return _bestPerItemError;
            }
        }

        public bool IsAuencoderDataSet
        {
            get
            {
                return
                    _validationData.IsAuencoderDataSet;
            }
        }


        public MetricErrorValidation(
            IMetrics errorMetrics,
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount,
            int domainCountThreshold = 100)
        {
            if (errorMetrics == null)
            {
                throw new ArgumentNullException("errorMetrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            if (validationData.IsAuencoderDataSet)
            {
                throw new ArgumentException("Для этого валидатора годятся только датасеты НЕ для автоенкодера");
            }

            _errorMetrics = errorMetrics;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
            _domainCountThreshold = domainCountThreshold;
            _outputLength = validationData.Data[0].OutputLength;
        }

        public void Validate(
            MultiLayerNeuralNetwork network,
            string epocheRoot,
            float cumulativeError,
            bool allowToSave)
        {
            var netResults = network.ComputeOutput(_validationData.GetInputPart());

            var totalError = 0f;
            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                var err = _errorMetrics.Calculate(
                    netResult,
                    testItem.Output);

                totalError += err;
            }

            var perItemError = totalError/_validationData.Count;

            Console.WriteLine(
                "Validation per-item error: {0}",
                perItemError);

            #region сохраняем картинки

            if (_validationData.IsAbleToVisualize)
            {
                if (_validationData.IsAuencoderDataSet)
                {
                    _validationData.SaveAsGrid(
                        Path.Combine(epocheRoot, "grid.bmp"),
                        netResults.Take(_visualizeAsGridCount).ToList());

                    //со случайного индекса
                    var startIndex = (int)((DateTime.Now.Millisecond/1000f)*(_validationData.Count - _visualizeAsPairCount));

                    var pairList = new List<Pair<float[], float[]>>();
                    for (var cc = startIndex; cc < startIndex + _visualizeAsPairCount; cc++)
                    {
                        var i = new Pair<float[], float[]>(
                            _validationData[cc].Input,
                            netResults[cc]);
                        pairList.Add(i);
                    }
                    _validationData.SaveAsPairList(
                        Path.Combine(epocheRoot, "reconstruct.bmp"),
                        pairList);
                }
            }

            #endregion

            #region если результат лучше, чем был, то сохраняем его

            if (_bestPerItemError >= perItemError)
            {
                _bestPerItemError = Math.Min(_bestPerItemError, perItemError);

                if (allowToSave)
                {
                    var networkFilename = string.Format(
                        "{0}-err={1}.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        perItemError);

                    SerializationHelper.SaveToFile(
                        network,
                        Path.Combine(epocheRoot, networkFilename));

                    Console.WriteLine("Saved!");
                }
            }

            #endregion
        }

    }

}
