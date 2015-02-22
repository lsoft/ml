using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron.Factory
{
    public interface INeuronFactory
    {
        INeuron CreateInputNeuron(int thisIndex);

        INeuron CreatePseudoNeuron(
            );

        INeuron CreateTrainableNeuron(
            int weightCount
            );
    }
}
