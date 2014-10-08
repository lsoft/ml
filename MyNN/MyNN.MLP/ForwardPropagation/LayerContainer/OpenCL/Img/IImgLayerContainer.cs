using OpenCL.Net.Wrapper.Mem.Img;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Img
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