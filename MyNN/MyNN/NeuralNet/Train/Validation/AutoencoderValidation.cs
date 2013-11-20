using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Train.Validation
{
    public class AutoencoderValidation : IValidation
    {
        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;
        private readonly int _outputLength;

        private float _bestCumulativeError = float.MaxValue;
        public float BestCumulativeError
        {
            get
            {
                return _bestCumulativeError;
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

        public AutoencoderValidation(
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount)
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            if (!validationData.IsAuencoderDataSet)
            {
                throw new ArgumentException("Для этого валидатора годятся только датасеты для автоенкодера");
            }

            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
            _outputLength = validationData.Data[0].OutputLength;
        }

        public void Validate(
            MultiLayerNeuralNetwork network,
            string epocheRoot,
            float cumulativeError,
            bool allowToSave)
        {
            #region сохраняем картинки

            if (_validationData.IsAbleToVisualize)
            {
                if (_validationData.IsAuencoderDataSet)
                {
                    var netResults = network.ComputeOutput(_validationData.GetInputPart());

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

            if (_bestCumulativeError >= cumulativeError)
            {
                _bestCumulativeError = Math.Min(_bestCumulativeError, cumulativeError);

                if (allowToSave)
                {
                    var networkFilename = string.Format(
                        "{0}-cumulativeError={1}.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        cumulativeError);

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
