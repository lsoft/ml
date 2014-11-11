using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.WeightMask.Factory
{
    public interface IOpenCLWeightMaskContainerFactory
    {
        IOpenCLWeightMaskContainer CreateContainer(
            long arraySize
            );
    }
}
