using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Common.Data;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.LearningRateController;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine
{
    public interface IRBM
    {
        void Train(
            ITrainDataProvider trainDataProvider,
            IDataSet validationData,
            ILearningRate learningRateController,
            IAccuracyController accuracyController,
            int batchSize,
            int maxGibbsChainLength
            );
    }
}