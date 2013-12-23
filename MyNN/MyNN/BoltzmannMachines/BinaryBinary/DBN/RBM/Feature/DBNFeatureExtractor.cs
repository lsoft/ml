using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature
{
    public class DBNFeatureExtractor : IFeatureExtractor
    {
        private readonly int _layerIndex;

        private readonly int _featureCount;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly Bitmap _bitmap;

        private int _currentIndex = 0;
        private readonly int _q;

        public DBNFeatureExtractor(
            int layerIndex,
            int featureCount,
            int imageWidth,
            int imageHeight)
        {
            _layerIndex = layerIndex;

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
            if (_layerIndex == 0)
            {

                CreateContrastEnhancedBitmapFromLayer(
                    (_currentIndex%_q)*_imageWidth,
                    ((int) (_currentIndex/_q))*_imageHeight,
                    data);

                _currentIndex++;
            }
            else
            {
                if (_currentIndex == 0)
                {
                    var rectf = new RectangleF(70, 90, 90, 50);

                    var g = Graphics.FromImage(_bitmap);

                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    g.DrawString("no features", new Font("Tahoma", 12), Brushes.Black, rectf);

                    g.Flush();
                }

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
