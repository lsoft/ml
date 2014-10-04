using MyNN.MLP2.ForwardPropagation.Classic;
using OpenCL.Net.Wrapper.Mem.Img;

namespace MyNN.MLP2.ForwardPropagation.LayerContainer.OpenCL.Img
{
    public interface IImgLayerContainer : ILayerContainer
    {
        IntensityFloatImg WeightMem
        {
            get;
        }

        IntensityFloatImg NetMem
        {
            get;
        }

        IntensityFloatImg StateMem
        {
            get;
        }
    }
}