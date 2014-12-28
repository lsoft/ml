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
    public interface IBackpropagationFactory
    {
        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="artifactContainer">Container that stores MLP and other related data</param>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="validationDataProvider">Validation provider</param>
        /// <param name="config">Learning algorithm config</param>
        /// <returns>Backpropagation algorithm</returns>
        IBackpropagation CreateBackpropagation(
            IRandomizer randomizer,
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config);
    }
}
