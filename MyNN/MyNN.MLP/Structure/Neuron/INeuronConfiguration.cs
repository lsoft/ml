namespace MyNN.MLP.Structure.Neuron
{
    public interface INeuronConfiguration
    {
        int WeightsCount
        {
            get;
        }

        bool IsBiasNeuron
        {
            get;
        }

    }
}
