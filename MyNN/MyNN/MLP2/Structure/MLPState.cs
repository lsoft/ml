using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Structure
{
    public class MLPState : IMLPState
    {
        public ILayerState[] State
        {
            get;
            private set;
        }

        public MLPState(ILayerState[] state)
        {
            State = state;
        }
    }
}