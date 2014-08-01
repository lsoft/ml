using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp
{
    public interface ILayerPropagator
    {
        float[] ComputeLayer(
            ILayer layer,
            float[] inputVector);
    }
}