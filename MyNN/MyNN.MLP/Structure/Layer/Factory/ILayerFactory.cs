using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer.Factory
{
    public interface ILayerFactory
    {
        ILayer CreateInputLayer(
            int totalNeuronCount
            );

        ILayer CreateLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount
            );
    }
}
