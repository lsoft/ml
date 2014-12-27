using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables CPU-OpenCL
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

        public string GetPreprocessHiddenKernelZeroSource(
            int groupSize
            )
        {
            var kernelText = @"
__kernel void PreprocessKernel0(
    __global read_only float * nextLayerDeDz,
    __global read_only float * nextLayerWeights,
    __global write_only float * gcache,

    int currentNeuronCount,
    int nextLayerNeuronCount
    )
{
    const int groupsizex = <GROUP_SIZE>;
    const int groupsizey = <GROUP_SIZE>;

    __local float cache[<GROUP_SIZE> * <GROUP_SIZE>];

    int globalx = get_global_id(0);
    int globaly = get_global_id(1);

    int groupx = get_group_id(0);
    int groupy = get_group_id(1);

    int ingrx = get_local_id(0);
    int ingry = get_local_id(1);

    int inCacheIndex = ingry * groupsizex + ingrx;
    cache[inCacheIndex] = 0;

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount && globaly < nextLayerNeuronCount)
    {
        int nextNeuronIndex = globaly;
        int nextWeightIndex = nextNeuronIndex* (currentNeuronCount + 1) + globalx;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        cache[inCacheIndex] = multiplied;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

    int current_local_size = groupsizey;
    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
    {
        if (ingry < offsety)
        {
            int other_index = ingry + offsety;
            if(other_index < current_local_size)
            {
                int readIndex = other_index * groupsizex + ingrx;
                int writeIndex = ingry * groupsizex + ingrx;

                cache[writeIndex] += cache[readIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        current_local_size = (current_local_size + 1) / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

            kernelText = kernelText.Replace("<GROUP_SIZE>", groupSize.ToString());

            return
                kernelText;
        }

        public string GetPreprocessHiddenKernelOneSource(
            )
        {
            var kernelText = @"
__kernel void PreprocessKernel1(
    __global float * gcache,

    int currentNeuronCount,
    int nextLayerNeuronCount,
    int groupsizex,
    int groupsizey,

    __local float * cache
    )
{
    int globalx = get_global_id(0);
    int globaly = get_global_id(1);

    int groupx = get_group_id(0);
    int groupy = get_group_id(1);

    int ingrx = get_local_id(0);
    int ingry = get_local_id(1);

    int inCacheIndex = ingry * groupsizex + ingrx;
    cache[inCacheIndex] = 0;

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount && globaly < nextLayerNeuronCount)
    {
        int nextNeuronIndex = globaly;
        int nextWeightIndex = nextNeuronIndex * currentNeuronCount + globalx;

//        float gvalue = gcache[nextWeightIndex];
//        gcache[nextWeightIndex] = 0;
//        cache[inCacheIndex] = gvalue;

         // 3 lines up is equivalent with one line below:

        cache[inCacheIndex] = atomic_xchg(gcache + nextWeightIndex, 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //фаза редукции

    int current_local_size = groupsizey;
    for(int offsety = (groupsizey + 1) / 2; offsety > 0; offsety = (offsety + (offsety > 1 ? 1 : 0)) / 2)
    {
        if (ingry < offsety)
        {
            int other_index = ingry + offsety;
            if(other_index < current_local_size)
            {
                int readIndex = other_index * groupsizex + ingrx;
                int writeIndex = ingry * groupsizex + ingrx;

                cache[writeIndex] += cache[readIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        current_local_size = (current_local_size + 1) / 2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //если группа не вылазит за пределы MLP
    if(globalx < currentNeuronCount)
    {
        //пишем в глобальный кеш
        gcache[groupy * currentNeuronCount + globalx] = cache[ingrx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

";

            return
                kernelText;
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