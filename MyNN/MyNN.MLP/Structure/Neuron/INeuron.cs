using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    public interface INeuron
    {
        float[] Weights
        {
            get;
        }

        float Bias
        {
            get;
            set;
        }

        INeuronConfiguration GetConfiguration();
    }
}