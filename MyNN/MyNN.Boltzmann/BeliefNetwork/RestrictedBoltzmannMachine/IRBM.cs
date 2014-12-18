using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSetProvider;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine
{
    public interface IRBM
    {
        void Train(
            IDataSetProvider trainDataProvider,
            IDataSet validationData,
            ILearningRate learningRateController,
            IAccuracyController accuracyController,
            int batchSize,
            int maxGibbsChainLength
            );
    }
}