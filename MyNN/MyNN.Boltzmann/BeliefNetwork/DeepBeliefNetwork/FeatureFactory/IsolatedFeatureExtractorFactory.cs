using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;

namespace MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.FeatureFactory
{
    public class IsolatedFeatureExtractorFactory : IFeatureExtractorFactory
    {
        public IFeatureExtractor CreateFeatureExtractor(
            int hiddenNeuronCount)
        {
            var featureExtractor = new IsolatedFeatureExtractor(
                hiddenNeuronCount,
                28,
                28);

            return featureExtractor;
        }
    }
}