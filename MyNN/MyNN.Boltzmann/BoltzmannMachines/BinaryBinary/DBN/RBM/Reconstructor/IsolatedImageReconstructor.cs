using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using MyNN.Common.Data;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor
{
    public class IsolatedImageReconstructor : IImageReconstructor
    {
        private readonly IDataSet _dataSet;
        private readonly int _reconstructedCount;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly Bitmap _bitmap;

        public IsolatedImageReconstructor(
            IDataSet dataSet,
            int reconstructedCount,
            int imageWidth,
            int imageHeight)
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            _dataSet = dataSet;
            _reconstructedCount = Math.Min(reconstructedCount, dataSet.Count);
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            _bitmap = new Bitmap(
                _imageWidth * 2,
                _imageHeight * _reconstructedCount);
        }

        public Bitmap GetReconstructedBitmap(
            int startDataItemIndexIntoDataSet,
            List<float[]> reconstructedDataList
            )
        {
            if (reconstructedDataList == null)
            {
                throw new ArgumentNullException("reconstructedDataList");
            }

            var currentIndex = 0;
            foreach (var pair in reconstructedDataList.ZipInequalLength(_dataSet.Skip(startDataItemIndexIntoDataSet)))
            {
                var reconstructedData = pair.Value1;
                var originalData = pair.Value2.Input;

                CreateContrastEnhancedBitmapFromLayer(
                    0,
                    currentIndex * _imageHeight,
                    originalData);

                CreateContrastEnhancedBitmapFromLayer(
                    _imageWidth,
                    currentIndex * _imageHeight,
                    reconstructedData);

                currentIndex++;
            }

            return
                _bitmap;
        }

        public int GetReconstructedImageCount()
        {
            return
                _reconstructedCount;
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
