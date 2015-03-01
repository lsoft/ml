using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayerConfiguration
    {
        IFunction LayerActivationFunction
        {
            get;
        }

        IDimension SpatialDimension
        {
            get;
        }

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
