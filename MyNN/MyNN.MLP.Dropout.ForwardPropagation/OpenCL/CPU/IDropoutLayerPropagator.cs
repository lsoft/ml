using MyNN.MLP.DropConnect.WeightMask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.CPU
{
    public interface IDropoutLayerPropagator : ILayerPropagator
    {
        IOpenCLWeightMaskContainer MaskContainer
        {
            get;
        }
    }
}