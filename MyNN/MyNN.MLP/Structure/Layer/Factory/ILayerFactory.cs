using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer.Factory
{
    public interface ILayerFactory
    {
        ILayer CreateInputLayer(int withoutBiasNeuronCount);

        ILayer CreateLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron
            );
    }
}
