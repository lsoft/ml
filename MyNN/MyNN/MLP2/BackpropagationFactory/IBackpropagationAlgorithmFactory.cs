using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.BackpropagationFactory
{
    /// <summary>
    /// Factory for backpropagation algorithm
    /// </summary>
    public interface IBackpropagationAlgorithmFactory
    {
        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="artifactContainer">Container that stores MLP and other related data</param>
        /// <param name="net">Trained MLP</param>
        /// <param name="validationDataProvider">Validation provider</param>
        /// <param name="config">Learning algorithm config</param>
        /// <returns>Backpropagation algorithm</returns>
        BackpropagationAlgorithm GetBackpropagationAlgorithm(
            IRandomizer randomizer,
            CLProvider clProvider,
            IArtifactContainer artifactContainer,
            IMLP net,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config);
    }
}
