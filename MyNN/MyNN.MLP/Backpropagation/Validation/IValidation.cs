using MyNN.Common.ArtifactContainer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Backpropagation.Validation
{
    public interface IValidation
    {
        /// <summary>
        /// Epoche validation
        /// </summary>
        /// <param name="forwardPropagation">Forward propagator</param>
        /// <param name="epocheNumber">Number of current epoche. Null if it's pretrain call.</param>
        /// <param name="epocheContainer">Container that stores MLP and other related data</param>
        /// <returns>A class that contains an information about MLP accuracy in terms of validation method</returns>
        IAccuracyRecord Validate(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IArtifactContainer epocheContainer
            );
    }
}
