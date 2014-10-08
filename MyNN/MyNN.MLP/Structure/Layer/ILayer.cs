using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayer
    {
        bool IsBiasNeuronExists
        {
            get;
        }

        int NonBiasNeuronCount
        {
            get;
        }

        INeuron[] Neurons
        {
            get;
        }

        IFunction LayerActivationFunction
        {
            get;
        }

        void AddBiasNeuron();

        void RemoveBiasNeuron();

        string GetLayerInformation();

        ILayerConfiguration GetConfiguration();
    }
}