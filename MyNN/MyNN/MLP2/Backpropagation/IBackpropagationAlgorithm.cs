using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.AccuracyRecord;

namespace MyNN.MLP2.Backpropagation
{
    public interface IBackpropagationAlgorithm
    {
        IAccuracyRecord Train(ITrainDataProvider trainDataProvider);
    }
}