using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.KNN;
using MyNN.MLP2.ForwardPropagation;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator.KNNTester
{
    public class DefaultKNNTester : IKNNTester
    {
        private readonly IKNearestFactory _kNearestFactory;
        private readonly IDataSet _trainData;
        private readonly IDataSet _validationData;
        private readonly int _neighborCount;

        public DefaultKNNTester(
            IKNearestFactory kNearestFactory,
            IDataSet trainData,
            IDataSet validationData,
            int neighborCount
            )
        {
            if (kNearestFactory == null)
            {
                throw new ArgumentNullException("kNearestFactory");
            }
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _kNearestFactory = kNearestFactory;
            _trainData = trainData;
            _validationData = validationData;
            _neighborCount = neighborCount;
        }

        public void Test(
            IForwardPropagation forwardPropagation,
            int takeIntoAccount,
            out int total,
            out int correct
            )
        {
            //просчитываем обучающее множество
            var trainOutputList = forwardPropagation.ComputeOutput(_trainData);

            var forknn = new List<DataItem>();
            for (var cc = 0; cc < trainOutputList.Count; cc++)
            {
                forknn.Add(
                    new DataItem(
                        trainOutputList[cc].Take(takeIntoAccount).ToArray(),
                        _trainData[cc].Output));
            }

            //инициализируем knn
            //var knn = new CPUOpenCLKNearest(new DataSet(forknn));
            var knn = _kNearestFactory.CreateKNearest(new DataSet(forknn));

            //просчитываем валидационное множество
            var validationList = forwardPropagation.ComputeOutput(_validationData);

            //проверяем валидационное множество
            correct = 0;
            total = 0;
            for (var index = 0; index < _validationData.Count; index++)
            {
                var classindex = knn.Classify(
                    validationList[index].Take(takeIntoAccount).ToArray(),
                    _neighborCount);

                if (classindex == _validationData[index].OutputIndex)
                {
                    correct++;
                }

                total++;
            }

            ConsoleAmbientContext.Console.WriteLine(
                "KNN TEST: total {0}, correct {1},  {2}%                ",
                total,
                correct,
                ((int)100 * correct / total));
        }

    }
}
