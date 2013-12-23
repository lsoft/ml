using System;
using MyNN.LearningRateController;

namespace MyNN.MLP2.LearningConfig
{
    public class LearningAlgorithmConfig : ILearningAlgorithmConfig
    {
        /// <summary>
        /// ������ �������� �������� ������������ �������� �������� �� ������ �����
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
        /// ��� �������������
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
            float minErrorChange)
        {
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }

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
