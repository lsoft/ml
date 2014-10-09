using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Structure
{
    public interface IMLPConfiguration
    {
        ILayerConfiguration[] Layers
        {
            get;
        }

    }
}
