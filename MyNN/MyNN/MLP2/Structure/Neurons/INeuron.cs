using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons
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
    }
}