using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.KernelText
{
    /// <summary>
    /// Kernel source provider for dropconnect backpropagation epoche trainer that enables GPU-OpenCL
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

        /*
        public string GetPreprocessHiddenKernelZeroSource(
            int groupSize
            )
        {
            var kernelText = @"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void PreprocessKernel0(
    __global read_only float * nextLayerDeDz,
    __global read_only float * nextLayerWeights,
    __global write_only float * results,

    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    __local float * local_accum

    )
{
    int neuronIndex = get_group_id(0);

    // ������� ��������� �������� �������� ����, �� ��������� �������� ������������ (with Kahan Algorithm)

    KahanAccumulator accDeDz = GetEmptyKahanAcc();
    for (
        int nextNeuronIndex = get_local_id(0);
        nextNeuronIndex < nextLayerNeuronCount; 
        nextNeuronIndex += get_local_size(0)
        )
    {
        int nextWeightIndex = 
            ComputeWeightIndex(currentLayerNeuronCount, nextNeuronIndex) + 
            neuronIndex;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        KahanAddElement(&accDeDz, multiplied);
    }

    local_accum[get_local_id(0)] = accDeDz.Sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(local_accum);
    barrier(CLK_LOCAL_MEM_FENCE);
    float currentDeDz = local_accum[0];

    if(get_local_id(0) == 0)
    {
        results[neuronIndex] = currentDeDz;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
";

            return
                kernelText;

        }
        //*/

        /*
        public string GetPreprocessHiddenKernelZeroSource(
            int groupSize
            )
        {
            return
                GetPreprocessHiddenKernelZeroSource_New(groupSize);
        }
        //*/

        //*/

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