using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;


namespace MyNN.MLP2.Structure.Layer
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
    }
}