using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.BoltzmannMachines;

namespace MyNN.BeliefNetwork.ImageReconstructor
{
    public class MockImageReconstructor : IImageReconstructor
    {
        public void AddPair(
            int dataItemIndexIntoDataSet, 
            float[] reconstructedData)
        {
            //nothing to do
        }

        public Bitmap GetReconstructedBitmap()
        {
            var result = new Bitmap(300, 100);

            using (var g = Graphics.FromImage(result))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawString(
                    "Dataset cannot be visualized",
                    new Font("Tahoma", 12),
                    Brushes.Black,
                    new RectangleF(0, 0, 300, 100)
                    );

                g.Flush();
            }

            return result;
        }

        public int GetReconstructedImageCount()
        {
            return 1;
        }
    }
}
