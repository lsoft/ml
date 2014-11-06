using System;
using MyNN.Common.LearningRateController;
using MyNN.MLP.Backpropagation.Metrics;

namespace MyNN.MLP.LearningConfig
{
    public class LearningAlgorithmConfig : ILearningAlgorithmConfig
    {
        /// <summary>
        /// Метрика, на которую обучается MLP
        /// </summary>
        public IMetrics TargetMetrics
        {
            get;
            private set;
        }

        /// <summary>
        /// Объект задающий динамику коэффициента скорости обучения от номера эпохи
        /// </summary>
        public ILearningRate LearningRateController
        {
            get;
            private set;
        }

        /// <summary>
        /// Size of the butch. -1 means fullbatch size. 
        /// </summary>
        public int BatchSize
        {
            get;
            private set;
        }

        public float RegularizationFactor
        {
            get;
            private set;
        }

        public int MaxEpoches
        {
            get;
            private set;
        }

        /// <summary>
        /// If cumulative error for all training examples is less then MinError, then algorithm stops 
        /// </summary>
        public float MinError
        {
            get;
            private set;
        }

        /// <summary>
        /// If cumulative error change for all training examples is less then MinErrorChange, then algorithm stops 
        /// </summary>
        public float MinErrorChange
        {
            get;
            private set;
        }

        /// <summary>
        /// Для сериализатора
        /// </summary>
        public LearningAlgorithmConfig()
        {
        }

        public LearningAlgorithmConfig(
            IMetrics targetMetrics,
            ILearningRate learningRateController,
            int batchSize,
            float regularizationFactor,
            int maxEpoches,
            float minError,
            float minErrorChange)
        {
            if (targetMetrics == null)
            {
                throw new ArgumentNullException("targetMetrics");
            }
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }

            TargetMetrics = targetMetrics;
            LearningRateController = learningRateController;
            BatchSize = batchSize;
            RegularizationFactor = regularizationFactor;
            MaxEpoches = maxEpoches;
            MinError = minError;
            MinErrorChange = minErrorChange;
        }

        public void ReassignBatchSize(int batchSize)
        {
            this.BatchSize = batchSize;
        }
    }
}
