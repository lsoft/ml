using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.OutputConsole;
using MyNN.KNN;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.AccuracyCalculator.KNNTester
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

            var forknn = new List<IDataItem>();
            //for (var cc = 0; cc < trainOutputList.Count; cc++)
            //{
            //    forknn.Add(
            //        new DataItem(
            //            trainOutputList[cc].Take(takeIntoAccount).ToArray(),
            //            _trainData.Data[cc].Output));

            //    //пускай здесь остается принудительно DataItem, так как вряд ли
            //    //будет реалистичный сценарий, когда будет эффективнее другой тип
            //    //датаитема в этом месте
            //}
            foreach (var pair in trainOutputList.ZipEqualLength(_trainData))
            {
                var trainOutputItem = pair.Value1;
                var trainItem = pair.Value2;

                forknn.Add(
                    new DataItem(
                        trainOutputItem.Take(takeIntoAccount).ToArray(),
                        trainItem.Output));

                //пускай здесь остается принудительно DataItem, так как вряд ли
                //будет реалистичный сценарий, когда будет эффективнее другой тип
                //датаитема в этом месте
            }

            //инициализируем knn
            //var knn = new KNearest(new DataSet(forknn));
            var knn = _kNearestFactory.CreateKNearest(forknn);

            //просчитываем валидационное множество
            var validationList = forwardPropagation.ComputeOutput(_validationData);

            //проверяем валидационное множество
            correct = 0;
            total = 0;
            //for (var index = 0; index < _validationData.Count; index++)
            //{
            //    var classindex = knn.Classify(
            //        validationList[index].Take(takeIntoAccount).ToArray(),
            //        _neighborCount);

            //    if (classindex == _validationData.Data[index].OutputIndex)
            //    {
            //        correct++;
            //    }

            //    total++;
            //}
            foreach (var pair in validationList.ZipEqualLength(_validationData))
            {
                var vli = pair.Value1;
                var vi = pair.Value2;

                var classindex = knn.Classify(
                    vli.Take(takeIntoAccount).ToArray(),
                    _neighborCount);

                if (classindex == vi.OutputIndex)
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
