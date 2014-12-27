using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation
{
    public interface ILayerContainer
    {
        void ClearAndPushNetAndState();

        void ReadInput(float[] data);
        
        void ReadWeightsFromLayer(ILayer layer);

        void PopNetAndState();
        
        ILayerState GetLayerState();

        void PopWeights();

        void WritebackWeightsToMLP(ILayer layer);
    }
}