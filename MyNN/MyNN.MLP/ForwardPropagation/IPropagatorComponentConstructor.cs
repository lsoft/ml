using MyNN.MLP.Structure;

namespace MyNN.MLP.ForwardPropagation
{
    public interface IPropagatorComponentConstructor
    {
        void CreateComponents(
            IMLP mlp,
            out ILayerContainer[] containers,
            out ILayerPropagator[] propagators
            );
    }
}