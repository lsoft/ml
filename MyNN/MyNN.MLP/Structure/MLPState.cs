using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Structure
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