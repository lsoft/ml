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

    public class NeuronConfiguration : INeuronConfiguration
    {
        public int WeightsCount
        {
            get;
            private set;
        }

        public bool IsBiasNeuron
        {
            get;
            private set;
        }

        public NeuronConfiguration(
            int weightsCount, 
            bool isBiasNeuron)
        {
            WeightsCount = weightsCount;
            IsBiasNeuron = isBiasNeuron;
        }
    }
}
