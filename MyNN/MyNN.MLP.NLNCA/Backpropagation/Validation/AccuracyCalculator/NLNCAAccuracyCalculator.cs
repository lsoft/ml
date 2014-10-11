using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.NLNCA.Backpropagation.Validation.AccuracyCalculator.KNNTester;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.AccuracyCalculator
{
    public class NLNCAAccuracyCalculator : IAccuracyCalculator
    {
        private readonly IKNNTester _knnTester;
        private readonly IDataSet _validationData;
        private readonly IArtifactContainer _artifactContainer;

        public NLNCAAccuracyCalculator(
            IKNNTester knnTester,
            IDataSet validationData,
            IArtifactContainer artifactContainer
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
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }

            _knnTester = knnTester;
            _validationData = validationData;
            _artifactContainer = artifactContainer;
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

            using (var s = _artifactContainer.GetWriteStreamForResource("knn_correct.csv"))
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