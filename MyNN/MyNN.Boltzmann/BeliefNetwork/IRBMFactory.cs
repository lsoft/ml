using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Boltzmann.BoltzmannMachines;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;

namespace MyNN.Boltzmann.BeliefNetwork
{
    public interface IRBMFactory
    {
        void CreateRBM(
            IDataSet trainData,
            IArtifactContainer rbmContainer,
            IImageReconstructor imageReconstructor,
            IFeatureExtractor featureExtractor,
            int visibleNeuronCount,
            int hiddenNeuronCount,
            out IRBM rbm,
            out IDataSetConverter forwardDataSetConverter,
            out IDataArrayConverter dataArrayConverter
            );
    }
}