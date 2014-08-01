using MyNN.MLP2.Structure.Layer;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two
{
    public interface ILayerMemContainer
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

        void ClearAndPushHiddenLayers();

        void PushInput(float[] data);

        void PushWeights(ILayer layer);

        void PopHiddenState();

        void PopLastLayerState();

        ILayerState GetLayerState();
    }
}