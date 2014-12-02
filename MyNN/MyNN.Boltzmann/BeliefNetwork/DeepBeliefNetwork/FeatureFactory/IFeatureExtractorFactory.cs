using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;

namespace MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.FeatureFactory
{
    public interface IFeatureExtractorFactory
    {
        IFeatureExtractor CreateFeatureExtractor(int hiddenNeuronCount);
    }
}