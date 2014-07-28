using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;

namespace MyNN.BeliefNetwork
{
    public interface IFeatureExtractorFactory
    {
        IFeatureExtractor CreateFeatureExtractor(int hiddenNeuronCount);
    }
}