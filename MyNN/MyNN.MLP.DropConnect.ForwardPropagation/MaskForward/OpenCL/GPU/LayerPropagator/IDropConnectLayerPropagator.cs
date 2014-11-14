using MyNN.Mask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU.LayerPropagator
{
    public interface IDropConnectLayerPropagator : ILayerPropagator
    {
        IOpenCLMaskContainer MaskContainer
        {
            get;
        }
    }
}