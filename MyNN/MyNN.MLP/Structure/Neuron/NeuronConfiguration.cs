namespace MyNN.MLP.Structure.Neuron
{
    public class NeuronConfiguration : INeuronConfiguration
    {
        public int WeightsCount
        {
            get;
            private set;
        }

        public NeuronConfiguration(
            int weightsCount
            )
        {
            WeightsCount = weightsCount;
        }
    }
}