using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.ForwardPropagation.WeightMaskContainer2.Factory
{
    public interface IOpenCLWeightBitMaskContainer2Factory
    {
        IOpenCLWeightMaskContainer2 CreateContainer2(
            CLProvider clProvider,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration
            );
    }
}
