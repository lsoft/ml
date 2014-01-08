using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.ForwardPropagation;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagaion.Validation
{
    public class MetricErrorValidation : IValidation
    {
        private readonly ISerializationHelper _serialization;
        private readonly IMetrics _errorMetrics;
        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;
        private readonly int _domainCountThreshold;

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
            ISerializationHelper serialization,
            IMetrics errorMetrics,
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount,
            int domainCountThreshold = 100)
        {
            if (serialization == null)
            {
                throw new ArgumentNullException("serialization");
            }
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

            _serialization = serialization;
            _errorMetrics = errorMetrics;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
            _domainCountThreshold = domainCountThreshold;
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

            var totalError = 0f;
            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                var err = _errorMetrics.Calculate(
                    netResult.State,
                    testItem.Output);

                totalError += err;
            }

            var perItemError = totalError/_validationData.Count;

            ConsoleAmbientContext.Console.WriteLine(
                "Validation per-item error: {0}",
                perItemError);

            #region сохраняем картинки

            if (_validationData.IsAbleToVisualize)
            {
                if (_validationData.IsAuencoderDataSet)
                {
                    _validationData.SaveAsGrid(
                        Path.Combine(epocheRoot, "grid.bmp"),
                        netResults.ConvertAll(j => j.State).Take(_visualizeAsGridCount).ToList());

                    //со случайного индекса
                    var startIndex = (int)((DateTime.Now.Millisecond/1000f)*(_validationData.Count - _visualizeAsPairCount));

                    var pairList = new List<Pair<float[], float[]>>();
                    for (var cc = startIndex; cc < startIndex + _visualizeAsPairCount; cc++)
                    {
                        var i = new Pair<float[], float[]>(
                            _validationData[cc].Input,
                            netResults[cc].State);
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

                    _serialization.SaveToFile(
                        forwardPropagation.MLP,
                        Path.Combine(epocheRoot, networkFilename));

                    ConsoleAmbientContext.Console.WriteLine("Saved!");
                }
            }

            #endregion

            return perItemError;
        }

    }

}
