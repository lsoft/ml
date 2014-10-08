using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public class ClassificationAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IMetrics _errorMetrics;
        private readonly IDataSet _validationData;
        private readonly int _domainCountThreshold;
        private readonly int _outputLength;

        public ClassificationAccuracyCalculator(
            IMetrics errorMetrics,
            IDataSet validationData,
            int domainCountThreshold = 100
            )
        {
            if (errorMetrics == null)
            {
                throw new ArgumentNullException("errorMetrics");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _errorMetrics = errorMetrics;
            _validationData = validationData;
            _domainCountThreshold = domainCountThreshold;

            _outputLength = validationData.Data[0].OutputLength;
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

            var correctArray = new int[_outputLength];
            var totalArray = new int[_outputLength];

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            netResults = forwardPropagation.ComputeOutput(_validationData);

            var totalError = 0f;

            var iterationIndex = 0;
            foreach (var netResult in netResults)
            {
                var testItem = _validationData[iterationIndex++];

                #region суммируем ошибку

                var err = _errorMetrics.Calculate(
                    netResult.NState,
                    testItem.Output);

                totalError += err;

                #endregion

                #region вычисляем успешность классификации

                var correctIndex = testItem.OutputIndex;

                var success = false;

                //берем максимальный вес на выходных
                var max = netResult.NState.Max();
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

            accuracyRecord = new ClassificationAccuracyRecord(
                epocheNumber ?? 0,
                totalCount,
                totalCorrectCount,
                perItemError
                );

            //ети две нижние строки возможно можно взять из acuracyRecord!!! переделать!
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

        }
    }
}