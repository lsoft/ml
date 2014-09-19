using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.Container
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