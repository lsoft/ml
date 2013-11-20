using System;
using MyNN.LearningRateController;
using MyNN.NeuralNet.Train.Metrics;

namespace MyNN.NeuralNet.LearningConfig
{
    [Serializable]
    public class LearningAlgorithmConfig : ILearningAlgorithmConfig
    {
        /// <summary>
        /// Объект задающий динамику коэффициента скорости обучения от номера эпохи
        /// </summary>
        public ILearningRate LearningRateController { get; private set; }

        /// <summary>
        /// Size of the butch. -1 means fullbatch size. 
        /// </summary>
        public int BatchSize { get; private set; }

        public float RegularizationFactor { get; private set; }

        public int MaxEpoches { get; private set; }

        /// <summary>
        /// If cumulative error for all training examples is less then MinError, then algorithm stops 
        /// </summary>
        public float MinError { get; private set; }

        /// <summary>
        /// If cumulative error change for all training examples is less then MinErrorChange, then algorithm stops 
        /// </summary>
        public float MinErrorChange { get; private set; }

        /// <summary>
        /// Function to minimize
        /// </summary>
        public IMetrics ErrorFunction { get; private set; }

        /// <summary>
        /// Для сериализатора
        /// </summary>
        public LearningAlgorithmConfig()
        {
        }

        public LearningAlgorithmConfig(
            ILearningRate learningRateController,
            int batchSize,
            float regularizationFactor,
            int maxEpoches,
            float minError,
            float minErrorChange,
            IMetrics errorFunction)
        {
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }
            if (errorFunction == null)
            {
                throw new ArgumentNullException("errorFunction");
            }

            LearningRateController = learningRateController;
            BatchSize = batchSize;
            RegularizationFactor = regularizationFactor;
            MaxEpoches = maxEpoches;
            MinError = minError;
            MinErrorChange = minErrorChange;
            ErrorFunction = errorFunction;
        }

        public void ReassignBatchSize(int batchSize)
        {
            this.BatchSize = batchSize;
        }
    }
}
