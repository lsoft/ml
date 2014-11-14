using MyNN.Mask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.CPU.LayerPropagator
{
    public interface IDropConnectLayerPropagator : ILayerPropagator
    {
        IOpenCLMaskContainer MaskContainer
        {
            get;
        }
    }
}