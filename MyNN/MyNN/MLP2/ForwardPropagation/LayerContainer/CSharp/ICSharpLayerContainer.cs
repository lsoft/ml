using MyNN.MLP2.ForwardPropagation.Classic;

namespace MyNN.MLP2.ForwardPropagation.LayerContainer.CSharp
{
    public interface ICSharpLayerContainer : ILayerContainer
    {
        float[] WeightMem
        {
            get;
        }

        float[] NetMem
        {
            get;
        }

        float[] StateMem
        {
            get;
        }
    }
}