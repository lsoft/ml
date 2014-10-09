using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayerConfiguration
    {
        INeuronConfiguration[] Neurons
        {
            get;
        }

        bool IsBiasNeuronExists
        {
            get;
        }

        int NonBiasNeuronCount
        {
            get;
        }
    }
}
