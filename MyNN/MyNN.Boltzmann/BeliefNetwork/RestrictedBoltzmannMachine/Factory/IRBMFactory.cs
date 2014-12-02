using MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.Converter;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter;
using MyNN.Boltzmann.BoltzmannMachines;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Factory
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