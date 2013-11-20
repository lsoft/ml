using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.KNN;
using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Train.Validation.NLNCA
{
    public class NLNCAValidation : IValidation
    {
        private readonly DataSet _trainData;
        private readonly DataSet _validationData;
        private readonly int _neighborCount;

        private int _validationKNNEpocheNumber = 0;
        private readonly Color[] _colors;

        public int ValidationKNNEpocheNumber
        {
            get
            {
                return _validationKNNEpocheNumber;
            }
        }

        public bool IsAuencoderDataSet
        {
            get
            {
                return
                    _validationData.IsAuencoderDataSet;
            }
        }

        public NLNCAValidation(
            DataSet trainData,
            DataSet validationData,
            IColorProvider colorProvider,
            int neighborCount)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (colorProvider == null)
            {
                throw new ArgumentNullException("colorProvider");
            }

            _trainData = trainData;
            _validationData = validationData;
            _colors = colorProvider.GetColors();
            _neighborCount = neighborCount;
        }

        public void Validate(
            MultiLayerNeuralNetwork network,
            string epocheRoot,
            float cumulativeError,
            bool allowToSave)
        {
            #region чистка файловой системы

            var bmpRoot = Path.Combine(network.FolderName, "bitmaps");
            var knnCorrectRoot = Path.Combine(network.FolderName, "knn_correct.csv");

            if (_validationKNNEpocheNumber == 0)
            {
                if (Directory.Exists(bmpRoot))
                {
                    Directory.Delete(bmpRoot, true);
                }
                Directory.CreateDirectory(bmpRoot);

                File.Delete(knnCorrectRoot);
            }

            #endregion

            #region считаем вероятности

            //var validationData = TransformImages("mnist/trainingset", int.MaxValue).Skip(TrainCountForOneCategory * 11).Take(ValidationCountForOneCategory * 10).ToList();

            ////итоговый логгинг
            //var uzkii = net.ComputeOutput(validationData.ToList().ConvertAll(j => j.Input));

            //var dodfCalculator =
            //    new MyNN.NeuralNet.Train.Algo.NCA2D.DodfCalculator(validationData.ConvertAll(j => j.OutputIndex).ToArray(), uzkii);

            //dodfCalculator.DumpPi("_validation_pi.csv", epocheNumber == 0);

            #endregion

            #region knn

            var total = 0;
            var correct = KNNTest(
                network,
                network.Layers.Last().NonBiasNeuronCount, //без отдельных нейронов для кодирования нерелевантных для расстояния между классами фич
                _neighborCount,
                out total);

            File.AppendAllText(
                knnCorrectRoot,
                DateTime.Now.ToString() + ";" + correct.ToString() + "\r\n");

            #endregion

            var ntr = network.ComputeOutput(_validationData.GetInputPart());

            #region выгружаем в файл

            //for (var cc = 0; cc < ntr.Count; cc++)
            //{
            //    File.AppendAllText(
            //        "traindataresults/" + validation2DEpocheNumber.ToString() + ".csv",
            //        TrainData[cc].OutputIndex + ";" + ntr[cc][0].ToString() + ";" + ntr[cc][1].ToString() + ";\r\n");
            //}

            #endregion

            #region рисуем на картинке

            if (ntr[0].Length == 2)
            {
                //рисуем на картинке
                var maxx = ntr.Max(j => j[0]);
                var minx = ntr.Min(j => j[0]);
                var maxy = ntr.Max(j => j[1]);
                var miny = ntr.Min(j => j[1]);

                var imageWidth = 500;
                var imageHeight = 500;

                var bitmap = new Bitmap(imageWidth, imageHeight);
                var ii = 0;

                using (var g = Graphics.FromImage(bitmap))
                {
                    g.DrawString(
                        minx.ToString() + ";" + miny.ToString(),
                        new Font("Tahoma", 12),
                        Brushes.Black,
                        0, 0);

                    g.DrawString(
                        maxx.ToString() + ";" + maxy.ToString(),
                        new Font("Tahoma", 12),
                        Brushes.Black,
                        300, 450);

                    foreach (var netResult in ntr)
                    {
                        var ox = netResult[0];
                        var oy = netResult[1];

                        var x = (ox - minx) * (imageWidth - 1) / (maxx - minx);
                        var y = (oy - miny) * (imageHeight - 1) / (maxy - miny);

                        g.DrawRectangle(
                            new Pen(_colors[_validationData[ii].OutputIndex]),
                            (int)x, (int)y, 1, 1
                            );
                        g.DrawRectangle(
                            new Pen(_colors[_validationData[ii].OutputIndex]),
                            (int)x, (int)y, 2, 2
                            );
                        g.DrawRectangle(
                            new Pen(_colors[_validationData[ii].OutputIndex]),
                            (int)x, (int)y, 3, 3
                            );
                        ii++;
                    }
                }

                bitmap.Save(
                    Path.Combine(
                        bmpRoot,
                        _validationKNNEpocheNumber.ToString("D5") + ".bmp"));
            }

            #endregion

            var networkFilename = string.Format(
                "{0}  epoche={1}  knn correct={2} out of {3}.mynn",
                DateTime.Now.ToString("yyyyMMddHHmmss"),
                _validationKNNEpocheNumber,
                correct,
                total);

            SerializationHelper.SaveToFile(
                network,
                Path.Combine(epocheRoot, networkFilename));

            Console.WriteLine("Saved!");

            _validationKNNEpocheNumber++;
        }

        private int KNNTest(
            MultiLayerNeuralNetwork net,
            int takeIntoAccount,
            int neighborCount,
            out int total)
        {
            //просчитываем обучающее множество
            var trainList = net.ComputeOutput(_trainData.GetInputPart());
            var forknn = new List<DataItem>();
            for (var cc = 0; cc < trainList.Count; cc++)
            {
                forknn.Add(
                    new DataItem(
                        trainList[cc].Take(takeIntoAccount).ToArray(),
                        _trainData[cc].Output));
            }

            //инициализируем knn
            var knn = new KNearest(new DataSet(forknn));

            //просчитываем валидационное множество
            var validationList = net.ComputeOutput(_validationData.GetInputPart());

            //проверяем валидационное множество
            int correct = 0;
            total = 0;
            for (var index = 0; index < _validationData.Count; index++)
            {
                var classindex = knn.Classify(
                    validationList[index].Take(takeIntoAccount).ToArray(),
                    neighborCount);

                if (classindex == _validationData[index].OutputIndex)
                {
                    correct++;
                }

                total++;
            }

            Console.WriteLine(
                "KNN TEST: total {0}, correct {1},  {2}%", 
                total, 
                correct, 
                ((int)100 * correct / total));

            return correct;
        }

    }
}
