using System;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer with transposed weights that enables GPU-OpenCL
    /// </summary>
    public class GPUTransposeKernelConstructor
    {
        private readonly MLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        public GPUTransposeKernelConstructor(
            MLP mlp,
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

            _mlp = mlp;
            _config = config;
        }

        #region calculation kernels source

        internal string GetOverwriteCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
            var result = _calculationKernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] = n;
");

            result =
                result.Replace(
                    "<nabla_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        regularizationFactor * currentLayerWeights[currentWeightIndex4] / dataCount
"));

            result =
                result.Replace(
                    "<nabla_regularization2>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
            regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount
"));


            result =
                result.Replace("<weight_update>", @"
        nabla[nablaNeuronShift + weightIndex] = deltaWeight;
");


            result =
                result.Replace(
                    "<weight_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        + regularizationFactor * currentLayerWeights4 / dataCount
"));

//            result =
//                result.Replace(
//                    "<weight_regularization2>",
//                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
//                        ? "0"
//                        : @"
//        regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount
//"));

            return result;
        }

        internal string GetIncrementCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
            var result = _calculationKernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] += n;
");

            result =
                result.Replace(
                    "<nabla_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        regularizationFactor * currentLayerWeights[currentWeightIndex4] / dataCount
"));

            result =
                result.Replace(
                    "<nabla_regularization2>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
            regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount
"));

            result =
                result.Replace("<weight_update>", @"
        nabla[nablaNeuronShift + weightIndex] += deltaWeight;
");

            result =
                result.Replace(
                    "<weight_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        + regularizationFactor * currentLayerWeights4 / dataCount
"));

//            result =
//                result.Replace(
//                    "<weight_regularization2>",
//                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
//                        ? "0"
//                        : @"
//        regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount
//"));

            return result;
        }


        private const string _calculationKernelsSource = @"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}


__kernel void HiddenLayerTrain(
    __global float * currentLayerNET,

    __global float * previousLayerLastState,
    __global float * currentLayerLastState,
    __global float * currentLayerDeDz,
    __global float * nextLayerDeDz,

    __global float * currentLayerWeights,
    __global float * nextLayerWeights,
            
    __global float * nabla,

    int previousLayerNeuronCount,
    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount
    
    ,__local float * local_next_nabla
    )
{
//    int neuronIndex = get_global_id(0);
//
//    if(neuronIndex >= currentLayerNeuronCount)
//    {
//        return;
//    }

    for (uint neuronIndex = get_group_id(0); neuronIndex < currentLayerNeuronCount; neuronIndex += get_num_groups(0))
    {
        int nextWeightIndex = ComputeWeightIndex(nextLayerNeuronCount, neuronIndex) + get_local_id(0);

        //������� ��������� �������� �������� ����, �� ��������� �������� ������������
        float currentDeDz = 0;
        for (int nextNeuronIndex = get_local_id(0); nextNeuronIndex < nextLayerNeuronCount; nextWeightIndex += get_local_size(0), nextNeuronIndex += get_local_size(0))
        {
            float nextNabla = nextLayerDeDz[nextNeuronIndex];

            //int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex;
            float nextWeight = nextLayerWeights[nextWeightIndex];

            float multiplied = nextWeight * nextNabla;

            currentDeDz += multiplied;
        }

        local_next_nabla[get_local_id(0)] = currentDeDz;
        barrier(CLK_LOCAL_MEM_FENCE);

        WarpReductionToFirstElement(local_next_nabla);
//        Reduction0(local_next_nabla);
        barrier(CLK_LOCAL_MEM_FENCE);
        currentDeDz = local_next_nabla[0];
//*/


//        //������� ��������� �������� �������� ����, �� ��������� �������� ������������
//        float currentDeDz = 0;
//        for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
//        {
//            int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex;
//
//            float nextWeight = nextLayerWeights[nextWeightIndex];
//            float nextNabla = nextLayerDeDz[nextNeuronIndex];
//            float multiplied = nextWeight * nextNabla;
//
//            currentDeDz += multiplied;
//        }
//*/

        float nOut = currentLayerNET[neuronIndex];
        currentDeDz *= <firstDerivative_nOut>;
        currentLayerDeDz[neuronIndex] = currentDeDz;

        int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

        for (
            int currentWeightIndex = get_local_id(0);
            currentWeightIndex < previousLayerNeuronCount; 
            currentWeightIndex += get_local_size(0)
            )
        {
            float prevOut = previousLayerLastState[currentWeightIndex];

            float regularizationCoef = 0;//                                                                         <nabla_regularization2>;
            float coef = prevOut + regularizationCoef;
            float n = learningRate * currentDeDz * coef;

            <nabla_update> //nabla[currentNablaIndex + currentWeightIndex] = n;
        }
//*/
    }
}


__kernel void OutputLayerTrain(
    __global float * currentLayerNET,

    __global float * previousLayerLastState,
    __global float * currentLayerLastState,
    __global float * currentLayerDeDz,

    __global float * desiredOutput,

    __global float * currentLayerWeights,
            
    __global float * nabla,

    int previousLayerNeuronCountTotal,
    int currentLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount

    )
{
//    int neuronIndex = get_global_id(0);
//
//    if(neuronIndex >= currentLayerNeuronCount)
//    {
//        return;
//    }

    for (uint neuronIndex = get_group_id(0); neuronIndex < currentLayerNeuronCount; neuronIndex += get_num_groups(0))
    {
        float nOut = currentLayerNET[neuronIndex];

        float n =
            <firstDerivative_nOut>
            * (desiredOutput[neuronIndex] - currentLayerLastState[neuronIndex]);

        currentLayerDeDz[neuronIndex] = n;

        int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

/*
        for (
            int weightIndex = 0;
            weightIndex < previousLayerNeuronCountTotal; 
            weightIndex += 1
            )
        {
            float deltaWeight =
                learningRate *
                n *
                (previousLayerLastState[weightIndex] );//                                                   + <weight_regularization2>);

            <weight_update> //nabla[nablaNeuronShift + weightIndex] = deltaWeight;
        }
//*/


        for (
            int weightIndex = get_local_id(0);
            weightIndex < previousLayerNeuronCountTotal; 
            weightIndex += get_local_size(0)
            )
        {
            float deltaWeight =
                learningRate *
                n *
                (previousLayerLastState[weightIndex] );//                                                   + <weight_regularization2>);

            <weight_update> //nabla[nablaNeuronShift + weightIndex] = deltaWeight;
        }
//*/
    }
}
";

        #endregion

        #region update weight kernel source

        public const string UpdateWeightKernelSource = @"
__kernel void UpdateWeightKernel(
    __global float * currentLayerWeights,
    const __global float * nabla,
    const float batchSize
    ,const int totalNeuronCount
    ,const int totalWeightCount
    , const int totalValueCount
    )
{
    int gi = get_global_id(0);

//    if(gi >= totalValueCount)
//    {
//        return;
//    }

    currentLayerWeights[gi] += nabla[gi] / batchSize;

//   for (uint y = get_group_id(0); y < totalNeuronCount; y += get_num_groups(0))
//   {
//        const __global float* from_ = nabla + y * totalWeightCount;
//        __global float* to_ = currentLayerWeights + y * totalWeightCount;
//
//        for (uint i = get_local_id(0); i < totalWeightCount; i += get_local_size(0))
//        {
//            to_[i] += from_[i] / batchSize;
//        }
//
//    }
}
";

        #endregion

    }
}