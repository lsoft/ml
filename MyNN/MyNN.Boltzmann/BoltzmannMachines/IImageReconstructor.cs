using System.Collections.Generic;
using System.Drawing;

namespace MyNN.Boltzmann.BoltzmannMachines
{
    public interface IImageReconstructor
    {
        Bitmap GetReconstructedBitmap(
            int startDataItemIndexIntoDataSet,
            List<float[]> reconstructedData
            );
        
        int GetReconstructedImageCount();
    }
}
