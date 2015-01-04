using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer.Factory
{
    public interface ILayerFactory
    {
        ILayer CreateInputLayer(
            IDimension dimension
            );

        ILayer CreateFullConnectedLayer(
            IFunction activationFunction,
            IDimension dimension,
            int previousLayerNeuronCount
            );
    }
}
