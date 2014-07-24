using System.Drawing;

namespace MyNN.BoltzmannMachines
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
