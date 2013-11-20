using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Train.Validation
{
    public class ClassificationValidation : IValidation
    {
        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;
        private readonly int _domainCountThreshold;
        private readonly int _outputLength;


        private int _bestCorrectCount = int.MinValue;
        public int BestCorrectCount
        {
            get
            {
                return _bestCorrectCount;
            }
        }

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


        public ClassificationValidation(
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount,
            int domainCountThreshold = 100)
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            if (validationData.IsAuencoderDataSet)
            {
                throw new ArgumentException("Для этого валидатора годятся только датасеты НЕ для автоенкодера");
            }

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
            var correctArray = new int[_outputLength];
            var totalArray = new int[_outputLength];

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var netResults = network.ComputeOutput(_validationData.GetInputPart());

            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];
                var correctIndex = testItem.OutputIndex;

                var success = false;

                //берем максимальный вес на выходных
                var max = netResult.Max();
                if (max > 0) //если это не нуль, значит хоть что-то да распозналось
                {
                    //если таких (максимальных) весов больше одного, значит, сеть не смогла точно идентифицировать символ
                    if (netResult.Count(j => Math.Abs(j - max) < float.Epsilon) == 1)
                    {

                        //таки смогла, присваиваем результат
                        var recognizeIndex = netResult.ToList().FindIndex(j => Math.Abs(j - max) < float.Epsilon);

                        success = correctIndex == recognizeIndex;
                    }
                }

                totalArray[correctIndex]++;

                if (success)
                {
                    totalCorrectCount++;
                    correctArray[correctIndex]++;
                }
                else
                {
                    totalFailCount++;
                }
            }

            var totalCount = totalCorrectCount + totalFailCount;

            var correctPercentCount = ((int)(totalCorrectCount * 10000 / totalCount) / 100.0);

            Console.WriteLine(
                string.Format(
                    "Success: {0}, fail: {1}, total: {2}, success %: {3}",
                    totalCorrectCount,
                    totalFailCount,
                    totalCount,
                    correctPercentCount));

            #region по классам репортим

            if (correctArray.Length < _domainCountThreshold)
            {
                for (var cc = 0; cc < correctArray.Length; cc++)
                {
                    Console.Write(cc + ": ");

                    var p = (int)((correctArray[cc] / (double)totalArray[cc]) * 10000) / 100.0;
                    Console.Write(correctArray[cc] + "/" + totalArray[cc] + "(" + p + "%)");

                    if (cc != (correctArray.Length - 1))
                    {
                        Console.Write("   ");
                    }
                    else
                    {
                        Console.WriteLine(string.Empty);
                    }
                }

                #region определяем классы, в которых нуль распознаваний

                var zeroClasses = string.Empty;
                var index = 0;
                foreach (var c in correctArray)
                {
                    if (c == 0)
                    {
                        zeroClasses += index;
                    }

                    index++;
                }

                if (!string.IsNullOrEmpty(zeroClasses))
                {
                    Console.WriteLine("Not recognized: " + zeroClasses);
                }

                #endregion
            }
            else
            {
                Console.WriteLine("Too many domains, details output is disabled");
            }

            #endregion

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

            if ((_bestCorrectCount <= totalCorrectCount && totalCorrectCount > 0) || (correctArray.Length >= 100 && _bestCumulativeError >= cumulativeError))
            {
                _bestCorrectCount = Math.Max(_bestCorrectCount, totalCorrectCount);
                _bestCumulativeError = Math.Min(_bestCumulativeError, cumulativeError);

                if (allowToSave)
                {
                    var networkFilename = string.Format(
                        "{0}-{1}-{2}%-err={3}.mynn",
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        totalCorrectCount,
                        correctPercentCount,
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
