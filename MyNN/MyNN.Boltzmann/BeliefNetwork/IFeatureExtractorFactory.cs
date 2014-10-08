using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;

namespace MyNN.Boltzmann.BeliefNetwork
{
    public interface IFeatureExtractorFactory
    {
        IFeatureExtractor CreateFeatureExtractor(int hiddenNeuronCount);
    }
}