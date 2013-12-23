using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using MyNN.Data.Visualizer;

namespace MyNN.Data.TypicalDataProvider
{
    public class MNISTVisualizer : IVisualizer
    {
        private const int ImageWidth = 28;
        private const int ImageHeight = 28;

        public void SaveAsGrid(string filepath, List<float[]> data)
        {
            if (filepath == null)
            {
                throw new ArgumentNullException("filepath");
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

            bitmap.Save(filepath);
        }

        public void SaveAsPairList(string filepath, List<Pair<float[], float[]>> data)
        {
            if (filepath == null)
            {
                throw new ArgumentNullException("filepath");
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

            bitmap.Save(filepath);
        }
    }
}
