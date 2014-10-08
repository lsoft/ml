using MyNN.Common.Data.TrainDataProvider;
using MyNN.MLP.AccuracyRecord;

namespace MyNN.MLP.Backpropagation
{
    public interface IBackpropagationAlgorithm
    {
        IAccuracyRecord Train(ITrainDataProvider trainDataProvider);
    }
}