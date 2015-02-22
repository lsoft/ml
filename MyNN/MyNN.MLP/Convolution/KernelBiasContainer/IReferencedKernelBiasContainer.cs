using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.KernelBiasContainer
{
    public interface IReferencedKernelBiasContainer
    {
        IDimension SpatialDimension
        {
            get;
        }
        
        int Width
        {
            get;
        }

        int Height
        {
            get;
        }

        IReferencedSquareFloat Kernel
        {
            get;
        }

        float Bias
        {
            get;
        }

        void ReadFrom(
            IReferencedKernelBiasContainer source
            );

        float GetValueFromCoordSafely(
            int fromw,
            int fromh
            );

        void IncrementBy(
            IReferencedSquareFloat nablaContainer,
            float nablaBias,
            float batchSize
            );
    }
}