using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using MyNN.Common.Data.Visualizer;
using MyNN.Common.Other;

namespace MyNN.Common.Data.DataLoader
{
    public class MNISTVisualizer : IVisualizer
    {
        private const int ImageWidth = 28;
        private const int ImageHeight = 28;

        private readonly Bitmap _gridBitmap;
        private readonly Bitmap _pairBitmap; 
        private readonly int _q;
        
        private int _gridCurrentIndex;
        private int _pairCurrentIndex;

        public MNISTVisualizer(
            int dataCount
            )
        {
            _q = (int)Math.Ceiling(Math.Sqrt(dataCount));
            _gridBitmap = new Bitmap(
                _q * ImageWidth,
                _q * ImageHeight);

            _pairBitmap = new Bitmap(
                ImageWidth * 2 + 1,
                ImageHeight * dataCount);

            _gridCurrentIndex = 0;
            _pairCurrentIndex = 0;
        }

        public void VisualizeGrid(
            float[] data
            )
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            BitmapHelper.CreateContrastEnhancedBitmapFrom(
                _gridBitmap,
                (_gridCurrentIndex % _q) * ImageWidth,
                ((int)(_gridCurrentIndex / _q)) * ImageHeight,
                data);

            _gridCurrentIndex++;
        }

        public void VisualizePair(Pair<float[], float[]> data)
        {
            BitmapHelper.CreateContrastEnhancedBitmapFrom(
                _pairBitmap,
                0,
                _pairCurrentIndex * ImageHeight,
                data.First);

            BitmapHelper.CreateContrastEnhancedBitmapFrom(
                _pairBitmap,
                ImageWidth + 1,
                _pairCurrentIndex * ImageHeight,
                data.Second);

            _pairCurrentIndex++;
        }

        public void SaveGrid(Stream writeStream)
        {
            _gridBitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
        }

        public void SavePairs(Stream writeStream)
        {
            _pairBitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
        }


        /*
        public void SaveAsGrid(
            Stream writeStream,
            List<float[]> data
            )
        {
            if (writeStream == null)
            {
                throw new ArgumentNullException("writeStream");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var q = (int)Math.Ceiling(Math.Sqrt(data.Count));
            var bitmap = new Bitmap(
                q * ImageWidth,
                q * ImageHeight);

            var currentIndex = 0;
            foreach (var i in data)
            {
                BitmapHelper.CreateContrastEnhancedBitmapFrom(
                    bitmap,
                    (currentIndex % q) * ImageWidth,
                    ((int)(currentIndex / q)) * ImageHeight,
                    i);

                currentIndex++;
            }

            bitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
        }

        public void SaveAsPairList(
            Stream writeStream,
            List<Pair<float[], float[]>> data)
        {
            if (writeStream == null)
            {
                throw new ArgumentNullException("writeStream");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var bitmap = new Bitmap(
                ImageWidth * 2 + 1,
                ImageHeight * data.Count);

            for (var cc = 0; cc < data.Count; cc++)
            {
                var d = data[cc];

                BitmapHelper.CreateContrastEnhancedBitmapFrom(
                    bitmap,
                    0,
                    cc * ImageHeight,
                    d.First);

                BitmapHelper.CreateContrastEnhancedBitmapFrom(
                    bitmap,
                    ImageWidth + 1,
                    cc * ImageHeight,
                    d.Second);
            }

            bitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
        }
        //*/

    }
}
