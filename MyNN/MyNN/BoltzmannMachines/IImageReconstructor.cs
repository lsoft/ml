using System.Drawing;

namespace MyNN.BoltzmannMachines
{
    public interface IImageReconstructor
    {
        void AddPair(
            int indexof,
            //float[] originalData, 
            float[] reconstructedData);
        
        Bitmap GetReconstructedBitmap();
    }
}
