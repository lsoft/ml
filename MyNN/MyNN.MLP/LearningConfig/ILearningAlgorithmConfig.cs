using MyNN.Common.LearningRateController;
using MyNN.MLP.Backpropagation.Metrics;

namespace MyNN.MLP.LearningConfig
{
    public interface ILearningAlgorithmConfig
    {
        /// <summary>
        /// Метрика, на которую обучается MLP
        /// </summary>
        IMetrics TargetMetrics
        {
            get;
        }

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
        /// If cumulative error for all training examples is less than MinError, then algorithm stops 
        /// </summary>
        float MinError
        {
            get;
        }

        /// <summary>
        /// If cumulative error change for all training examples is less than MinErrorChange, then algorithm stops 
        /// </summary>
        float MinErrorChange
        {
            get;
        }

        void ReassignBatchSize(int batchSize);
    }
}