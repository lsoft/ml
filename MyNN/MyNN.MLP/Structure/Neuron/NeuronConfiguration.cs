namespace MyNN.MLP.Structure.Neuron
{
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