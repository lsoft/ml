using System.Drawing;

namespace MyNN.Boltzmann.BoltzmannMachines
{
    public interface IImageReconstructor
    {
        void AddPair(
            int dataItemIndexIntoDataSet,
            float[] reconstructedData);
        
        Bitmap GetReconstructedBitmap();
        
        int GetReconstructedImageCount();
    }
}
