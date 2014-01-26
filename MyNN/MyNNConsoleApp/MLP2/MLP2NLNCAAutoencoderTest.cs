using System;
using System.Collections.Generic;
using System.Linq;
using MyNN;
using MyNN.Data;
using MyNN.Data.TypicalDataProvider;
using MyNN.KNN;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.MLP2
{
    /// <summary>
    /// тестирования обученного nlnca-автоенкодера
    /// </summary>
    public class MLP2NLNCAAutoencoderTest
    {
        public static void Test()
        {
            var rndSeed = 8013947;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainDataSet = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                //"_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                200
                );
            trainDataSet.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var testDataSet = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                //"_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            testDataSet.Normalize();


            int takeIntoAccount = 50;
            int neighborCount = 3;

            var net = SerializationHelper.LoadFromFile<MLP>(
                "NLNCA Autoencoder20131129225357 MLP2/epoche 20/20131130195557-perItemError=0,0690959.mynn");

            net.AutoencoderCutTail();

            using (var clProvider = new CLProvider())
            {
                //создаем объект просчитывающий сеть
                var forward = new CPUForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    net,
                    clProvider);

                var trainOutput = forward.ComputeOutput(trainDataSet);

                var knnTrainData = new List<DataItem>();
                var nknnTrainData = new List<DataItem>();
                for (var cc = 0; cc < trainOutput.Count; cc++)
                {
                    knnTrainData.Add(
                        new DataItem(
                            trainOutput[cc].Take(takeIntoAccount).ToArray(),
                            trainDataSet[cc].Output));

                    nknnTrainData.Add(
                        new DataItem(
                            trainOutput[cc].Skip(takeIntoAccount).ToArray(),
                            trainDataSet[cc].Output));
                }

                //инициализируем knn
                var knn = new KNearest(new DataSet(knnTrainData));
                var nknn = new KNearest(new DataSet(nknnTrainData));

                var testOutput = forward.ComputeOutput(testDataSet);


                ////рисуем на картинке
                //var maxx = testOutput.Max(j => j.State[0]);
                //var minx = testOutput.Min(j => j.State[0]);
                //var maxy = testOutput.Max(j => j.State[1]);
                //var miny = testOutput.Min(j => j.State[1]);

                //var imageWidth = 500;
                //var imageHeight = 500;

                //var bitmap = new Bitmap(imageWidth, imageHeight);
                //var ii = 0;

                //using (var g = Graphics.FromImage(bitmap))
                //{
                //    var colors = new MNISTColorProvider().GetColors();

                //    foreach (var netResult in testOutput)
                //    {
                //        var ox = netResult.State[0];
                //        var oy = netResult.State[1];

                //        var x = (ox - minx) * (imageWidth - 1) / (maxx - minx);
                //        var y = (oy - miny) * (imageHeight - 1) / (maxy - miny);
                        
                //        g.DrawRectangle(
                //            new Pen(colors[testDataSet[ii].OutputIndex]),
                //            (int)x, (int)y, 1, 1
                //            );
                //        g.DrawRectangle(
                //            new Pen(colors[testDataSet[ii].OutputIndex]),
                //            (int)x, (int)y, 2, 2
                //            );
                //        g.DrawRectangle(
                //            new Pen(colors[testDataSet[ii].OutputIndex]),
                //            (int)x, (int)y, 3, 3
                //            );
                //        ii++;
                //    }

                //    g.DrawString(
                //        minx.ToString() + ";" + miny.ToString(),
                //        new Font("Tahoma", 12),
                //        Brushes.Black,
                //        0, 0);

                //    g.DrawString(
                //        maxx.ToString() + ";" + maxy.ToString(),
                //        new Font("Tahoma", 12),
                //        Brushes.Black,
                //        300, 450);

                //}

                //bitmap.Save(
                //    Path.Combine(
                //        ".",
                //        "_1.bmp"));


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

                    var classindexNotNLNCA = nknn.Classify(
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
