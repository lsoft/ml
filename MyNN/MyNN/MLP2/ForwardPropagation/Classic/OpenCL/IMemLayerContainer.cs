using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL
{
    public interface IMemLayerContainer : ILayerContainer
    {
        MemFloat WeightMem
        {
            get;
        }

        MemFloat NetMem
        {
            get;
        }

        MemFloat StateMem
        {
            get;
        }
    }
}