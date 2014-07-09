using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer
{
    /// <summary>
    /// Epoche trainer for backpropagation family algorithms
    /// </summary>
    public interface IBackpropagationEpocheTrainer
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
        void PreTrainInit(DataSet data);

        /// <summary>
        /// Do epoche training
        /// </summary>
        /// <param name="data">Train data</param>
        /// <param name="epocheRoot">Root folder for current epoche</param>
        /// <param name="learningRate">Learning rate coef</param>
        void TrainEpoche(
            DataSet data,
            string epocheRoot,
            float learningRate);
    }
}