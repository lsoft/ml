using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ForwardPropagation.Classic
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