using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation
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