using MyNN.Common.ArtifactContainer;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.BackpropagationFactory
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
