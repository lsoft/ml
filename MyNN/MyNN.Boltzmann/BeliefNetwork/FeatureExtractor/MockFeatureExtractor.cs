using System.Drawing;
using System.Drawing.Drawing2D;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;

namespace MyNN.Boltzmann.BeliefNetwork.FeatureExtractor
{
    public class MockFeatureExtractor : IFeatureExtractor
    {
        public void AddFeature(float[] data)
        {
            //nothing to do
        }

        public Bitmap GetFeatureBitmap()
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
    }
}