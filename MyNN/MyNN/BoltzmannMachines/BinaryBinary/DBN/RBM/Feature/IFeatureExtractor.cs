using System.Drawing;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature
{
    public interface IFeatureExtractor
    {
        void AddFeature(float[] data);
        Bitmap GetFeatureBitmap();
    }
}
