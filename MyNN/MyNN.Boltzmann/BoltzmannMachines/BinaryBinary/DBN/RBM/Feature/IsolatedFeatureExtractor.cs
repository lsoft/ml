using System;
using System.Drawing;
using System.Linq;

namespace MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature
{
    public class IsolatedFeatureExtractor : IFeatureExtractor
    {
        private readonly int _featureCount;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly Bitmap _bitmap;

        private int _currentIndex = 0;
        private readonly int _q;

        public IsolatedFeatureExtractor(
            int featureCount,
            int imageWidth,
            int imageHeight)
        {
            _featureCount = featureCount;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;

            _q = (int)Math.Ceiling(Math.Sqrt(featureCount));
            _bitmap = new Bitmap(
                _q * _imageWidth,
                _q * _imageHeight);
        }

        public void AddFeature(float[] data)
        {
            if (_currentIndex < _featureCount)
            {
                CreateContrastEnhancedBitmapFromLayer(
                    (_currentIndex%_q)*_imageWidth,
                    ((int) (_currentIndex/_q))*_imageHeight,
                    data);

                _currentIndex++;
            }
        }

        public Bitmap GetFeatureBitmap()
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
