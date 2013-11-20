using MyNN.LearningRateController;
using MyNN.NeuralNet.Train.Metrics;

namespace MyNN.NeuralNet.LearningConfig
{
    public interface ILearningAlgorithmConfig
    {
        /// <summary>
        /// Объект задающий динамику коэффициента скорости обучения от номера эпохи
        /// </summary>
        ILearningRate LearningRateController
        {
            get;
        }

        /// <summary>
        /// Size of the butch. -1 means fullbatch size. 
        /// </summary>
        int BatchSize
        {
            get;
        }

        float RegularizationFactor
        {
            get;
        }

        int MaxEpoches
        {
            get;
        }

        /// <summary>
        /// If cumulative error for all training examples is less then MinError, then algorithm stops 
        /// </summary>
        float MinError
        {
            get;
        }

        /// <summary>
        /// If cumulative error change for all training examples is less then MinErrorChange, then algorithm stops 
        /// </summary>
        float MinErrorChange
        {
            get;
        }

        /// <summary>
        /// Function to minimize
        /// </summary>
        IMetrics ErrorFunction
        {
            get;
        }

        void ReassignBatchSize(int batchSize);
    }
}