using MyNN.Mask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL
{
    public interface IDropoutLayerPropagator : ILayerPropagator
    {
        IOpenCLMaskContainer MaskContainer
        {
            get;
        }

        int MaskShift
        {
            get;
        }
    }
}