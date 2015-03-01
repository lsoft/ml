using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables CPU-OpenCL
    /// </summary>
    public class KernelTextProvider : IKernelTextProvider
    {
        private readonly IKernelTextProvider _kp;
        
        public KernelTextProvider(
            ILearningAlgorithmConfig config
            )
        {
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
    __global float * nablaWeights,
    int weightCount, //общее количество флоатов для обработки (для всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount, //количество флоатов для обработки ОДНИМ кернелом (должно быть кратно 4м!!!)
    float batchSize,
    __global float * currentLayerBiases,
    __global float * nablaBiases,
    int biasesCount
)
{
    int kernelIndex = get_global_id(0);
    
    int d1StartIndex = kernelIndex * kernelDataCount;
    int d1Count = min(kernelDataCount, weightCount - d1StartIndex);

    int d4StartIndex = d1StartIndex / 4;
    int d4Count = d1Count / 4;
    
    int d1StartRemainder = d1StartIndex + d4Count * 4;

    for(int cc = d4StartIndex; cc < d4StartIndex + d4Count; cc++)
    {
        float4 currentLayerWeights4 = vload4(cc, currentLayerWeights);
        float4 nabla4 = vload4(cc, nablaWeights);

        float4 result = currentLayerWeights4 + nabla4 / batchSize;

        vstore4(
            result,
            cc,
            currentLayerWeights);
    }

    for(int cc = d1StartRemainder; cc < d1StartIndex + d1Count; cc++)
    {
        currentLayerWeights[cc] += nablaWeights[cc] / batchSize;
    }

    if(get_global_id(0) == 0)
    {
        for(int cc = 0; cc < biasesCount; cc++)
        {
            currentLayerBiases[cc] += nablaBiases[cc] / batchSize;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
";
            }
        }

        #endregion

    }
}