namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public interface ICSharpLayerContainer : ILayerContainer
    {
        float[] WeightMem
        {
            get;
        }

        float[] BiasMem
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