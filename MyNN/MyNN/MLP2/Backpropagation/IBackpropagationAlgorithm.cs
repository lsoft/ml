using MyNN.Data.TrainDataProvider;

namespace MyNN.MLP2.Backpropagation
{
    public interface IBackpropagationAlgorithm
    {
        void Train(ITrainDataProvider trainDataProvider);
    }
}