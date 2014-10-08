using System.Drawing;
using System.Drawing.Drawing2D;
using MyNN.Boltzmann.BoltzmannMachines;

namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor
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
                    Brushes.White,
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
