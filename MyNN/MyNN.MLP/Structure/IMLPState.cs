using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Structure
{
    public interface IMLPState
    {
        ILayerState[] LState
        {
            get;
        }
    }
}
