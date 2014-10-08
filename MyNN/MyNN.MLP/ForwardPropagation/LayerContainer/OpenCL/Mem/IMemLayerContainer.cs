using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem
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