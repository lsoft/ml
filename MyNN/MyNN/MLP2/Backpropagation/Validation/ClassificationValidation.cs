using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Saver;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation
{
    public class ClassificationValidation : IValidation
    {
        private readonly IMLPSaver _mlpSaver;
        private readonly IMetrics _errorMetrics;
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

        public bool IsAuencoderDataSet
        {
            get
            {
                return
                    _validationData.IsAuencoderDataSet;
            }
        }


        public ClassificationValidation(
            IMLPSaver mlpSaver,
            IMetrics errorMetrics,
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount,
            int domainCountThreshold = 100)
        {
            if (mlpSaver == null)
            {
                throw new ArgumentNullException("mlpSaver");
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

            _mlpSaver = mlpSaver;
            _errorMetrics = errorMetrics;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
            _domainCountThreshold = domainCountThreshold;
            _outputLength = validationData.Data[0].OutputLength;
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
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }

            var correctArray = new int[_outputLength];
            var totalArray = new int[_outputLength];

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            var totalError = 0f;

            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                #region суммируем ошибку

                var err = _errorMetrics.Calculate(
                    netResult.State,
                    testItem.Output);

                totalError += err;

                #endregion

                #region вычисляем успешность классификации

                var correctIndex = testItem.OutputIndex;

                var success = false;

                //берем максимальный вес на выходных
                var max = netResult.State.Max();
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

                #endregion
            }

            var perItemError = totalError / _validationData.Count;
            var totalCount = totalCorrectCount + totalFailCount;

            var correctPercentCount = ((int)(totalCorrectCount * 10000 / totalCount) / 100.0);

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Error = {0}, per-item error = {1}",
                    totalError,
                    DoubleConverter.ToExactString(perItemError)));

            ConsoleAmbientContext.Console.WriteLine(
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
                    ConsoleAmbientContext.Console.Write(cc + ": ");

                    var p = (int)((correctArray[cc] / (double)totalArray[cc]) * 10000) / 100.0;
                    ConsoleAmbientContext.Console.Write(correctArray[cc] + "/" + totalArray[cc] + "(" + p + "%)");

                    if (cc != (correctArray.Length - 1))
                    {
                        ConsoleAmbientContext.Console.Write("   ");
                    }
                    else
                    {
                        ConsoleAmbientContext.Console.WriteLine(string.Empty);
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
                    ConsoleAmbientContext.Console.WriteLine("Not recognized: " + zeroClasses);
                }

                #endregion
            }
            else
            {
                ConsoleAmbientContext.Console.WriteLine("Too many domains, details output is disabled");
            }

            #endregion

            #region сохраняем картинки

            if (_validationData.IsAbleToVisualize)
            {
                if (_validationData.IsAuencoderDataSet)
                {
                    _validationData.SaveAsGrid(
                        Path.Combine(epocheRoot, "grid.bmp"),
                        netResults.ConvertAll(j => j.State).Take(_visualizeAsGridCount).ToList());

                    //со случайного индекса
                    var startIndex = (int)((DateTime.Now.Millisecond / 1000f) * (_validationData.Count - _visualizeAsPairCount));

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

            if (_bestCorrectCount <= totalCorrectCount && totalCorrectCount > 0)
            {
                _bestCorrectCount = Math.Max(_bestCorrectCount, totalCorrectCount);

                if (allowToSave)
                {
                    var accuracyRecord = new AccuracyRecord(
                        totalCount,
                        totalCorrectCount);

                    _mlpSaver.Save(
                        epocheRoot,
                        accuracyRecord,
                        forwardPropagation.MLP);

                    ConsoleAmbientContext.Console.WriteLine("Saved!");
                }
            }

            #endregion

            return perItemError;
        }

    }

}
