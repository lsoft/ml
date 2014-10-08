using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.Backpropagation.EpocheTrainer
{
    /// <summary>
    /// Epoche trainer for backpropagation family algorithms
    /// </summary>
    public interface IEpocheTrainer
    {
        /// <summary>
        /// Forward propagator
        /// </summary>
        IForwardPropagation ForwardPropagation
        {
            get;
        }

        /// <summary>
        /// Primary init
        /// </summary>
        /// <param name="data">Train dataset</param>
        void PreTrainInit(IDataSet data);

        /// <summary>
        /// Do epoche training
        /// </summary>
        /// <param name="data">Train data</param>
        /// <param name="artifactContainer">Container that stores MLP and other related data</param>
        /// <param name="learningRate">Learning rate coef</param>
        void TrainEpoche(
            IDataSet data,
            IArtifactContainer artifactContainer,
            float learningRate);
    }
}