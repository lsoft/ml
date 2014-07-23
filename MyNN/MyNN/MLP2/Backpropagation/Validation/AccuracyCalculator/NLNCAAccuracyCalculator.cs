using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.KNN;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator.KNNTester;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure.Layer;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator
{
    public class NLNCAAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IKNNTester _knnTester;
        private readonly IDataSet _validationData;
        private readonly IMLPContainer _mlpContainer;

        public NLNCAAccuracyCalculator(
            IKNNTester knnTester,
            IDataSet validationData,
            IMLPContainer mlpContainer
            )
        {
            if (knnTester == null)
            {
                throw new ArgumentNullException("knnTester");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (mlpContainer == null)
            {
                throw new ArgumentNullException("mlpContainer");
            }

            _knnTester = knnTester;
            _validationData = validationData;
            _mlpContainer = mlpContainer;
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            out List<ILayerState> netResults,
            out IAccuracyRecord accuracyRecord)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            var total = 0;
            var correct = 0;
            _knnTester.Test(
                forwardPropagation,
                forwardPropagation.MLP.Layers.Last().NonBiasNeuronCount, //��� ��������� �������� ��� ����������� ������������� ��� ���������� ����� �������� ���
                out total,
                out correct
                );

            netResults = forwardPropagation.ComputeOutput(_validationData);

            var result = new ClassificationAccuracyRecord(
                epocheNumber ?? 0,
                total,
                correct,
                float.MaxValue
                );

            using (var s = _mlpContainer.GetWriteStreamForResource("knn_correct.csv"))
            {
                var writeinfo = DateTime.Now + ";" + result.CorrectCount + "\r\n";
                var writebytes = Encoding.Unicode.GetBytes(writeinfo);

                s.Write(writebytes, 0, writebytes.Length);

                s.Flush();
            }

            accuracyRecord = result;
        }
    }
}