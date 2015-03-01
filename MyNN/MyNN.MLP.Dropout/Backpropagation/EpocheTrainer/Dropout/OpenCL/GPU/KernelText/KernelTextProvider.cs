using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU.KernelText
{
    /// <summary>
    /// Kernel source provider for dropout backpropagation epoche trainer that enables GPU-OpenCL
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
                    config
                    );
            }
            else
            {
                _kp = new KernelTextProviderWithoutRegularization(
                    config
                    );
            }
        }

        #region calculation kernels source

        public string GetOverwriteCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            )
        {
            return
                _kp.GetOverwriteCalculationKernelsSource(layerConfiguration);
        }

        public string GetIncrementCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            )
        {
            return
                _kp.GetIncrementCalculationKernelsSource(layerConfiguration);
        }

        #endregion

        #region update weight kernel source

        public string UpdateWeightKernelSource
        {
            get
            {
                return @"
__kernel void UpdateWeightKernel(
    __global float * currentLayerWeights,
    const __global float * nablaWeights,
    const float batchSize,
    const int weightCount,
    __global float * currentLayerBiases,
    const __global float * nablaBiases,
    const int biasesCount
    )
{
    int gi = get_global_id(0);

    float wshift = nablaWeights[gi] / batchSize;
    currentLayerWeights[gi] += wshift;

    if(gi < biasesCount)
    {
        float bshift = nablaBiases[gi] / batchSize;
        currentLayerBiases[gi] += bshift;
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
";
            }
        }

        #endregion

    }
}