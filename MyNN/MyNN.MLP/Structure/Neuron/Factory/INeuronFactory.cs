using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron.Factory
{
    public interface INeuronFactory
    {
        INeuron CreateBiasNeuron();

        INeuron CreateInputNeuron(int thisIndex);

        INeuron CreateTrainableNeuron(
            IFunction activationFunction,
            int weightCount);
    }
}
