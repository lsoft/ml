using System;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public class ClassificationAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IMetrics _errorMetrics;
        private readonly IDataSet _validationData;
        private readonly int _domainCountThreshold;
        private readonly int _outputLength;
        
        private readonly AccuracyCalculatorBatchIterator _batchIterator;

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

            _outputLength = validationData.OutputLength;

            _batchIterator = new AccuracyCalculatorBatchIterator();
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IDrawer drawer, 
            out IAccuracyRecord accuracyRecord
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            //drawer allowed to be null

            if (drawer != null)
            {
                drawer.SetSize(
                    _validationData.Count
                    );
            }

            var correctArray = new int[_outputLength];
            var totalArray = new int[_outputLength];

            var totalCorrectCount = 0;
            var totalFailCount = 0;

            var totalErrorAcc = new KahanAlgorithm.Accumulator();

            _batchIterator.IterateByBatch(
                _validationData,
                forwardPropagation,
                (netResult, testItem) =>
                {
                    #region ������ ����

                    if (drawer != null)
                    {
                        drawer.DrawItem(netResult);
                    }

                    #endregion

                    #region ��������� ������

                    var err = _errorMetrics.Calculate(
                        testItem.Output,
                        netResult.NState
                        );

                    KahanAlgorithm.AddElement(ref totalErrorAcc, err);

                    #endregion

                    #region ��������� ���������� �������������

                    var correctIndex = testItem.OutputIndex;

                    var success = false;

                    //����� ������������ ��� �� ��������
                    var max = netResult.NState.Max();
                    if (max > 0) //���� ��� �� ����, ������ ���� ���-�� �� ������������
                    {
                        //���� ����� (������������) ����� ������ ������, ������, ���� �� ������ ����� ���������������� ������
                        if (netResult.Count(j => Math.Abs(j - max) < float.Epsilon) == 1)
                        {
                            //���� ������, ����������� ���������
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
                });

            var totalError = totalErrorAcc.Sum;

            var perItemError = totalError / _validationData.Count;
            var totalCount = totalCorrectCount + totalFailCount;

            var correctPercentCount = ((int)((long)totalCorrectCount * 10000 / totalCount) / 100.0);

            accuracyRecord = new ClassificationAccuracyRecord(
                epocheNumber ?? 0,
                totalCount,
                totalCorrectCount,
                perItemError
                );

            //��� ��� ������ ������ �������� ����� ����� �� acuracyRecord!!! ����������!
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

            #region �� ������� ��������

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

                #region ���������� ������, � ������� ���� �������������

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

            if (drawer != null)
            {
                drawer.Save();
            }
        }

    }
}