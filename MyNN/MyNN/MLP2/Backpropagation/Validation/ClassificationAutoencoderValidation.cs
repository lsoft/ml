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
    public class ClassificationAutoencoderValidation : IValidation
    {
        private readonly IMLPSaver _mlpSaver;
        private readonly IMetrics _metrics;
        private readonly DataSet _validationData;
        private readonly int _visualizeAsGridCount;
        private readonly int _visualizeAsPairCount;
        private readonly int _domainCountThreshold;

        private readonly int _classificationLength;
        private readonly int _autoencoderLength;

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
                    false; //другие типы DataSet этот класс просто не примет (он применим только к ClassificationAuencoderDataSet)
            }
        }

        public bool IsClassificationAuencoderDataSet
        {
            get
            {
                return
                    true; //другие типы DataSet этот класс просто не примет (он применим только к ClassificationAuencoderDataSet)
            }
        }

        public ClassificationAutoencoderValidation(
            IMLPSaver mlpSaver,
            IMetrics metrics,
            DataSet validationData,
            int visualizeAsGridCount,
            int visualizeAsPairCount,
            int domainCountThreshold = 100)
        {
            if (mlpSaver == null)
            {
                throw new ArgumentNullException("mlpSaver");
            }
            if (metrics == null)
            {
                throw new ArgumentNullException("metrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (validationData.IsAuencoderDataSet)
            {
                throw new ArgumentException("AuencoderDataSet не применим к данному классу, рассмотрите AutoencoderValidation");
            }
            if (validationData.IsClassificationAuencoderDataSet)
            {
                throw new ArgumentException("Не надо делать ClassificationAuencoder-датасет, сам сделаю");
            }

            _mlpSaver = mlpSaver;
            _metrics = metrics;
            _validationData = validationData;
            _visualizeAsGridCount = visualizeAsGridCount;
            _visualizeAsPairCount = visualizeAsPairCount;
            _domainCountThreshold = domainCountThreshold;

            _autoencoderLength = validationData.Data[0].Input.Length;
            _classificationLength = validationData.Data[0].OutputLength;
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

            var netResults = forwardPropagation.ComputeOutput(_validationData);

            int totalCount;
            int correctCount;
            ExecuteClassificationValidation(
                netResults.ConvertAll(j => j.State.GetSubArray(0, _classificationLength)),
                out totalCount,
                out correctCount);

            ExecuteAutoencoderValidation(
                forwardPropagation,
                epocheRoot,
                allowToSave,
                netResults.ConvertAll(j => j.State.GetSubArray(_classificationLength, _autoencoderLength)));


            //считаем ошибку

            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<DataItem>(_validationData.Count + 1);
            for (var i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DataItem(
                        _validationData[i].Output.Concatenate(_validationData[i].Input),
                        netResults[i].State));
            }

            var totalError = d.AsParallel().Sum(j => _metrics.Calculate(j.Input, j.Output));

            var perItemError = totalError / _validationData.Count;

            #region если результат лучше, чем был, то сохраняем его

            if (_bestPerItemError >= perItemError)
            {
                _bestPerItemError = Math.Min(_bestPerItemError, perItemError);

                if (allowToSave)
                {
                    var accuracyRecord = new AccuracyRecord(
                        perItemError,
                        totalCount,
                        correctCount);

                    _mlpSaver.Save(
                        epocheRoot,
                        accuracyRecord,
                        forwardPropagation.MLP);

                    ConsoleAmbientContext.Console.WriteLine("Saved!");
                }
            }

            #endregion

            if (!allowToSave)
            {
                ConsoleAmbientContext.Console.WriteLine(
                    "Per item error = {0} (summary)",
                    perItemError);
            }

            return
                perItemError;
        }

        private void ExecuteAutoencoderValidation(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave,
            List<float[]> netResults)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }
            if (netResults == null)
            {
                throw new ArgumentNullException("netResults");
            }

            #region сохраняем картинки

            //преобразуем в вид, когда в DataItem.Input - правильный ВЫХОД (обучаемый выход),
            //а в DataItem.Output - РЕАЛЬНЫЙ выход, а их разница - ошибка обучения
            var d = new List<DataItem>(_validationData.Count + 1);
            for (int i = 0; i < _validationData.Count; i++)
            {
                d.Add(
                    new DataItem(
                        _validationData[i].Input,
                        netResults[i]));
            }

            var totalError = d.AsParallel().Sum(j => _metrics.Calculate(j.Input, j.Output));

            var perItemError = totalError / _validationData.Count;

            if (_validationData.IsAbleToVisualize)
            {
                _validationData.SaveAsGrid(
                    Path.Combine(epocheRoot, "grid.bmp"),
                    netResults.Take(_visualizeAsGridCount).ToList());

                //со случайного индекса
                var startIndex = (int) ((DateTime.Now.Millisecond/1000f)*(_validationData.Count - _visualizeAsPairCount));

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

            #endregion
        }

        private void ExecuteClassificationValidation(
            List<float[]> netResults,
            out int totalCount,
            out int correctCount)
        {
            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var correctArray = new int[_classificationLength];
            var totalArray = new int[_classificationLength];

            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                #region вычисляем успешность классификации

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

                #endregion
            }

            totalCount = totalCorrectCount + totalFailCount;
            correctCount = totalCorrectCount;

            var correctPercentCount = ((int) (totalCorrectCount*10000/totalCount)/100.0);

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

                    var p = (int) ((correctArray[cc]/(double) totalArray[cc])*10000)/100.0;
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
        }
    }

}
