using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.CPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables CPU-OpenCL
    /// </summary>
    internal class KernelTextProvider : IKernelTextProvider
    {
        private readonly IKernelTextProvider _kp;
        
        public KernelTextProvider(
            IMLP mlp,
            ILearningAlgorithmConfig config)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            if (Math.Abs(config.RegularizationFactor) >= float.Epsilon)
            {
                _kp = new KernelTextProviderWithRegularization(
                    mlp,
                    config);
            }
            else
            {
                _kp = new KernelTextProviderWithoutRegularization(
                    mlp,
                    config);
            }
        }

        #region calculation kernels source

        public string GetOverwriteCalculationKernelsSource(int layerIndex)
        {
            return
                _kp.GetOverwriteCalculationKernelsSource(layerIndex);
        }

        public string GetIncrementCalculationKernelsSource(int layerIndex)
        {
            return
                _kp.GetIncrementCalculationKernelsSource(layerIndex);
        }

        #endregion

        #region update weight kernel source

        public string UpdateWeightKernelSource
        {
            get
            {
                return
                    _kp.UpdateWeightKernelSource;
            }
        }

        #endregion

    }
}