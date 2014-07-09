using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Autoencoders
{
    public interface IAutoencoder
    {
        IMLP Train(
            ILearningAlgorithmConfig config, 
            ITrainDataProvider trainDataProvider,
            IValidation validation);
    }
}