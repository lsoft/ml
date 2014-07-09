using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.KNN;
using MyNN.MLP2.ForwardPropagation;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation.NLNCA
{
    public class NLNCAValidation : IValidation
    {
        private readonly IKNearestFactory _kNearestFactory;
        private readonly ISerializationHelper _serialization;
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
            IKNearestFactory kNearestFactory,
            ISerializationHelper serialization,
            DataSet trainData,
            DataSet validationData,
            IColorProvider colorProvider,
            int neighborCount)
        {
            if (kNearestFactory == null)
            {
                throw new ArgumentNullException("kNearestFactory");
            }
            if (serialization == null)
            {
                throw new ArgumentNullException("serialization");
            }
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

            _kNearestFactory = kNearestFactory;
            _serialization = serialization;
            _trainData = trainData;
            _validationData = validationData;
            _colors = colorProvider.GetColors();
            _neighborCount = neighborCount;
        }

        public float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            #region чистка файловой системы

            var bmpRoot = Path.Combine(forwardPropagation.MLP.FolderName, "bitmaps");
            var knnCorrectRoot = Path.Combine(forwardPropagation.MLP.FolderName, "knn_correct.csv");

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

            #region knn

            var total = 0;
            var correct = KNNTest(
                forwardPropagation,
                forwardPropagation.MLP.Layers.Last().NonBiasNeuronCount, //без отдельных нейронов для кодирования нерелевантных для расстояния между классами фич
                _neighborCount,
                out total);

            File.AppendAllText(
                knnCorrectRoot,
                DateTime.Now.ToString() + ";" + correct.ToString() + "\r\n");

            #endregion

            var ntr = forwardPropagation.ComputeOutput(_validationData);

            #region выгружаем в файл

            //for (var cc = 0; cc < ntr.Count; cc++)
            //{
            //    File.AppendAllText(
            //        "traindataresults/" + validation2DEpocheNumber.ToString() + ".csv",
            //        TrainData[cc].OutputIndex + ";" + ntr[cc][0].ToString() + ";" + ntr[cc][1].ToString() + ";\r\n");
            //}

            #endregion

            #region рисуем на картинке

            if (ntr[0].State.Length == 2)
            {
                //рисуем на картинке
                var maxx = ntr.Max(j => j.State[0]);
                var minx = ntr.Min(j => j.State[0]);
                var maxy = ntr.Max(j => j.State[1]);
                var miny = ntr.Min(j => j.State[1]);

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
                        var ox = netResult.State[0];
                        var oy = netResult.State[1];

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

            _serialization.SaveToFile(
                forwardPropagation.MLP,
                Path.Combine(epocheRoot, networkFilename));

            ConsoleAmbientContext.Console.WriteLine("Saved!");

            _validationKNNEpocheNumber++;

            //return perItemError;
            return float.MaxValue;
        }

        private int KNNTest(
            IForwardPropagation forwardPropagation,
            int takeIntoAccount,
            int neighborCount,
            out int total)
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

            ConsoleAmbientContext.Console.WriteLine(
                "KNN TEST: total {0}, correct {1},  {2}%                ",
                total,
                correct,
                ((int) 100*correct/total));

            return correct;
        }

    }
}
