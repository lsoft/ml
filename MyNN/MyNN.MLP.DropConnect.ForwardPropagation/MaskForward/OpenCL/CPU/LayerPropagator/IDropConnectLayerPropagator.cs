using MyNN.MLP.DropConnect.WeightMask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.CPU.LayerPropagator
{
    public interface IDropConnectLayerPropagator : ILayerPropagator
    {
        IOpenCLWeightMaskContainer MaskContainer
        {
            get;
        }
    }
}