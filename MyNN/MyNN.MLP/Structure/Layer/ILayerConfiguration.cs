using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayerConfiguration
    {
        INeuronConfiguration[] Neurons
        {
            get;
        }

        int TotalNeuronCount
        {
            get;
        }

        int WeightCount
        {
            get;
        }

        int BiasCount
        {
            get;
        }
    }
}
