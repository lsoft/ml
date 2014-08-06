using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Structure
{
    public class MLPState : IMLPState
    {
        public ILayerState[] LState
        {
            get;
            private set;
        }

        public MLPState(ILayerState[] lState)
        {
            LState = lState;
        }
    }
}