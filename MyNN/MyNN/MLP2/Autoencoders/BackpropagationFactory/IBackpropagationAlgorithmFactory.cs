using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.Autoencoders.BackpropagationFactory
{
    public interface IBackpropagationAlgorithmFactory
    {
        BackpropagationAlgorithm GetBackpropagationAlgorithm(
            IRandomizer randomizer,
            CLProvider clProvider,
            MLP net,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config);
    }
}
