using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two
{
    public interface ILayerContainer
    {
        void ClearAndPushHiddenLayers();

        void PushInput(float[] data);
        
        void PushWeights(ILayer layer);
        
        void PopHiddenState();
        
        void PopLastLayerState();
        
        ILayerState GetLayerState();
    }
}