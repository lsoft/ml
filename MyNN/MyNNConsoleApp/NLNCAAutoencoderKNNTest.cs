using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.KNN;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Train.Algo;

namespace MyNNConsoleApp
{
    /// <summary>
    /// тестирования обученного nlnca-автоенкодера
    /// </summary>
    public class NLNCAAutoencoderKNNTest
    {
        public static void Test(
            DataSet trainDataSet, 
            DataSet testDataSet, 
            int takeIntoAccount, 
            int neighborCount)
        {
            if (trainDataSet == null)
            {
                throw new ArgumentNullException("trainDataSet");
            }
            if (testDataSet == null)
            {
                throw new ArgumentNullException("testDataSet");
            }

            var net = SerializationHelper.LoadFromFile<MultiLayerNeuralNetwork>(
                "MLP20131123211604/epoche 92/20131125091943-cumulativeError=3,402823E+38.mynn");
                //"MLP20131123211604/epoche 0/20131123213846-cumulativeError=3,402823E+38.mynn");
            net.AutoencoderCut();

            using (var universe = new VNNCLProvider(net))
            {
                //создаем объект просчитывающий сеть
                var computer =
                    new VOpenCLComputer(universe, true);

                net.SetComputer(computer);

                var trainOutput = net.ComputeOutput(trainDataSet.GetInputPart());
                var trainData = new List<DataItem>();
                for (var cc = 0; cc < trainOutput.Count; cc++)
                {
                    trainData.Add(
                        new DataItem(
                            trainOutput[cc].Take(takeIntoAccount).ToArray(),
                            trainDataSet[cc].Output));
                }

                //инициализируем knn
                var knn = new KNearest(new DataSet(trainData));

                var testOutput = net.ComputeOutput(testDataSet.GetInputPart());

                //проверяем валидационное множество
                int total = 0;
                int correctNLNCA = 0;
                int correctNotNLNCA = 0;
                for (var index = 0; index < testDataSet.Count; index++)
                {
                    var classindexNLNCA = knn.Classify(
                        testOutput[index].Take(takeIntoAccount).ToArray(),
                        neighborCount);

                    if (classindexNLNCA == testDataSet[index].OutputIndex)
                    {
                        correctNLNCA++;
                    }

                    var classindexNotNLNCA = knn.Classify(
                        testOutput[index].Skip(takeIntoAccount).ToArray(),
                        neighborCount);

                    if (classindexNotNLNCA == testDataSet[index].OutputIndex)
                    {
                        correctNotNLNCA++;
                    }

                    total++;
                }

                Console.WriteLine(
                    "KNN TEST: total {0}, NLNCA-correct = ({1}, {2}%), NotNLNCA-correct = ({3}, {4}%)",
                    total,
                    correctNLNCA,
                    ((int)100 * correctNLNCA / total),
                    correctNotNLNCA,
                    ((int)100 * correctNotNLNCA / total)
                    );
            }

        }
    }
}
