using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
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

        private readonly AccuracyCalculatorBatchIterator _batchIterator;

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

            _batchIterator = new AccuracyCalculatorBatchIterator();
        }

        public void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IDrawer drawer,
            out IAccuracyRecord accuracyRecord)
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

            var total = 0;
            var correct = 0;
            _knnTester.Test(
                forwardPropagation,
                forwardPropagation.MLP.Layers.Last().TotalNeuronCount, //без отдельных нейронов для кодирования нерелевантных для расстояния между классами фич
                out total,
                out correct
                );

            //var netResults = forwardPropagation.ComputeOutput(_validationData);

            //foreach (var netResult in netResults)
            //{
            //    #region рисуем итем

            //    if (drawer != null)
            //    {
            //        drawer.DrawItem(netResult);
            //    }

            //    #endregion
            //}

            _batchIterator.IterateByBatch(
                _validationData,
                forwardPropagation,
                (netResult, testItem) =>
                {
                    #region рисуем итем

                    if (drawer != null)
                    {
                        drawer.DrawItem(netResult);
                    }

                    #endregion
                });

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

            if (drawer != null)
            {
                drawer.Save();
            }
        }
    }
}