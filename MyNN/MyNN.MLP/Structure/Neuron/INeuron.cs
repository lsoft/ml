using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    public interface INeuron
    {
        float[] Weights
        {
            get;
        }

        IFunction ActivationFunction
        {
            get;
        }

        bool IsBiasNeuron
        {
            get;
        }

        INeuronConfiguration GetConfiguration();
    }
}