using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables GPU-OpenCL
    /// </summary>
    public class KernelTextProvider : IKernelTextProvider
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