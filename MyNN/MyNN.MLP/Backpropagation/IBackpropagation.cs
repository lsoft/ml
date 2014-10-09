using MyNN.Common.Data.TrainDataProvider;
using MyNN.MLP.AccuracyRecord;

namespace MyNN.MLP.Backpropagation
{
    public interface IBackpropagation
    {
        IAccuracyRecord Train(ITrainDataProvider trainDataProvider);
    }
}