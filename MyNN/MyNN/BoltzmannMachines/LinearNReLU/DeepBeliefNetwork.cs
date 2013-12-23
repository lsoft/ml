using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.BoltzmannMachines.LinearNReLU.RBM;
using MyNN.Data;
using MyNN.MLP2.Randomizer;


namespace MyNN.BoltzmannMachines.LinearNReLU
{
    public class DeepBeliefNetwork : IImageReconstructor
    {
        private readonly int[] _layerSizes;

        private readonly IRandomizer _randomizer;

        private int _imageWidth,
                    _imageHeight;

        private DataSet _trainData;
        private DataSet _validationData;

        public string Name
        {
            get;
            private set;
        }

        private int _layerIndex;
        private List<RestrictedBoltzmannMachine> _rbmList;
        private Bitmap _bitmap;
        private int _currentIndex;

        public DeepBeliefNetwork(
            IRandomizer randomizer,
            int imageWidth,
            int imageHeight,
            params int[] layerSizes)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (layerSizes == null || layerSizes.Length < 2)
            {
                throw new ArgumentException("layerSizes");
            }

            _randomizer = randomizer;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _layerSizes = layerSizes;

            this.Name = "DBN" + DateTime.Now.ToString("yyyyMMddHHmmss");
        }

        public void Train(
            string rootFolder,
            DataSet trainData,
            DataSet validationData,
            int maxGibbsChainLength,
            float learningRage,
            int epocheCount,
            int reconstructedCount)
        {
            _trainData = trainData;
            _validationData = validationData;

            _layerIndex = 0;
            _currentIndex = 0;
            var reallyReconstructedCount = Math.Min(reconstructedCount, validationData.Count);
            _bitmap = new Bitmap(
                _imageWidth * 2,
                _imageHeight * reallyReconstructedCount);

            var artifactFolderRoot = string.IsNullOrEmpty(rootFolder) ? this.Name : rootFolder;
            Directory.CreateDirectory(artifactFolderRoot);

            var currentTrainData = trainData;
            var currentValidationData = validationData;

            _rbmList = new List<RestrictedBoltzmannMachine>();
            for (var cc = 0; cc < _layerSizes.Length - 1; cc++)
            {
                var rbmFolder = Path.Combine(
                    artifactFolderRoot,
                    "rbm_layer" + cc);

                var rbm = new RestrictedBoltzmannMachine(
                    _randomizer,
                    _layerSizes[cc],
                    _layerSizes[cc + 1],
                    _imageWidth,
                    _imageHeight
                    );
                _rbmList.Add(rbm);

                rbm.Train(
                    rbmFolder,
                    currentTrainData,
                    currentValidationData,
                    maxGibbsChainLength,
                    learningRage,
                    epocheCount,
                    this,
                    reconstructedCount,
                    cc == 0,
                    cc == 0);

                _layerIndex++;

                //семплируем данные для обучения для следующей rbm
                var currentTrainData2 = new List<DataItem>();
                foreach (var di in currentTrainData)
                {
                    var hidden = new float[_layerSizes[cc + 1]];

                    rbm.SampleHiddenLayer(
                        di.Input,
                        hidden);

                    currentTrainData2.Add(
                        new DataItem(
                            hidden,
                            di.Output));
                }
                currentTrainData = new DataSet(
                    currentTrainData2);

                //семплируем данные для валидации для следующей rbm
                var currentValidationData2 = new List<DataItem>();
                foreach (var di in currentValidationData)
                {
                    var hidden = new float[_layerSizes[cc + 1]];

                    rbm.SampleHiddenLayer(
                        di.Input,
                        hidden);

                    currentValidationData2.Add(
                        new DataItem(
                            hidden,
                            di.Output));
                }
                currentValidationData = new DataSet(
                    currentValidationData2);
            }

        }

        #region Implementation of IImageReconstructor

        public void AddPair(
            int indexof,
            float[] reconstructedData)
        {
            for (var cc = _layerIndex - 1; cc >= 0; cc--)
            {
                var visibleData = new float[this._rbmList[cc].VisibleNeuronCount];
                this._rbmList[cc].ComputeVisibleLayer(
                    reconstructedData,
                    visibleData);

                reconstructedData = visibleData;
            }

            var originalData = _validationData[indexof].Input;

            CreateContrastEnhancedBitmapFromLayer(
                0,
                _currentIndex * _imageHeight,
                originalData);

            CreateContrastEnhancedBitmapFromLayer(
                _imageWidth,
                _currentIndex * _imageHeight,
                reconstructedData);

            _currentIndex++;
        }


        public Bitmap GetReconstructedBitmap()
        {
            _currentIndex = 0;

            return
                _bitmap;
        }


        private void CreateContrastEnhancedBitmapFromLayer(
            int left,
            int top,
            float[] layer)
        {
            var max = layer.Take(_imageWidth * _imageHeight).Max(val => val);
            var min = layer.Take(_imageWidth * _imageHeight).Min(val => val);

            if (Math.Abs(min - max) <= float.Epsilon)
            {
                min = 0;
                max = 1;
            }

            for (int x = 0; x < _imageWidth; x++)
            {
                for (int y = 0; y < _imageHeight; y++)
                {
                    var value = layer[PointToIndex(x, y, _imageWidth)];
                    value = (value - min) / (max - min);
                    var b = (byte)Math.Max(0, Math.Min(255, value * 255.0));

                    _bitmap.SetPixel(left + x, top + y, Color.FromArgb(b, b, b));
                }
            }
        }

        private int PointToIndex(int x, int y, int width)
        {
            return y * width + x;
        }
        #endregion
    }
}
