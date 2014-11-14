using System.Security.Cryptography.X509Certificates;
using MyNN.Mask;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.CPU
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