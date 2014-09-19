using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using OpenCL.Net.Wrapper.Mem.Img;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPUIMG.Container
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