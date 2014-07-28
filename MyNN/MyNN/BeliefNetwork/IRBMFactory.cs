using MyNN.BeliefNetwork.ImageReconstructor;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BoltzmannMachines;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Data;
using MyNN.Data.DataSetConverter;
using MyNN.MLP2.ArtifactContainer;

namespace MyNN.BeliefNetwork
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