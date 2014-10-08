using System.Drawing;

namespace MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature
{
    public interface IFeatureExtractor
    {
        void AddFeature(float[] data);
        Bitmap GetFeatureBitmap();
    }
}
