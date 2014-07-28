using System;
using MyNN.BeliefNetwork.Accuracy;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.LearningRateController;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine
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