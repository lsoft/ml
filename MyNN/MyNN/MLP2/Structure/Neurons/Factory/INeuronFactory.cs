using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons.Factory
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
