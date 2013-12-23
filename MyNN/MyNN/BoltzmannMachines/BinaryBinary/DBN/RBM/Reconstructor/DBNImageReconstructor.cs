using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using MyNN.Data;


namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor
{
    public class DBNImageReconstructor : IImageReconstructor
    {
        private readonly DeepBeliefNetwork _dbn;
        private readonly int _layerIndex;
        private readonly DataSet _trainData;

        private readonly int _reconstructedCount;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly Bitmap _bitmap;

        private int _currentIndex = 0;

        public DBNImageReconstructor(
            DeepBeliefNetwork dbn,
            int layerIndex,
            DataSet trainData,
            int reconstructedCount,
            int imageWidth,
            int imageHeight)
        {
            _dbn = dbn;
            _layerIndex = layerIndex;
            _trainData = trainData;

            _reconstructedCount = Math.Min(reconstructedCount, trainData.Count);
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            _bitmap = new Bitmap(
                _imageWidth * 2,
                _imageHeight * _reconstructedCount);
        }

        public void AddPair(
            int indexof,
            //float[] originalData, 
            float[] reconstructedData)
        {
            //var r = new Random(888);
            for (var cc = _layerIndex - 1; cc >= 0; cc--)
            {
                //originalData = originalData.ToList().ConvertAll(j => (float)Math.Round(j)).ToArray();
                //reconstructedData = reconstructedData.ToList().ConvertAll(j => j < r.NextDouble() ? 0f : 1f).ToArray();

                //originalData = _dbn.RBMList[cc].ComputeVisibleFromHidden(originalData);
                reconstructedData = _dbn.RBMList[cc].ComputeVisibleFromHidden(reconstructedData);
            }

            var originalData = _trainData[indexof].Input;

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
    }
}
