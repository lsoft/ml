using System;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.TransposedClassic2.OpenCL.GPU
{
    /// <summary>
    /// Kernel source provider.
    /// </summary>
    public class KernelConstructor
    {
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        public KernelConstructor(
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

            _mlp = mlp;
            _config = config;
        }

        #region calculation kernels source

        internal string GetOverwriteCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
            var result = CalculationKernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] = n;
");

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
                    "<weight_regularization2>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount
"));

            return result;
        }

        internal string GetIncrementCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
            var result = CalculationKernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] += n;
");

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
                    "<weight_regularization2>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
        regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount
"));

            return result;
        }


        private const string CalculationKernelsSource = @"
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
    float dataCount,

    __local float * local_accum
    )
{
    int neuronIndex = get_group_id(0);

    if(neuronIndex < currentLayerNeuronCount)
    {
        int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

        KahanAccumulator accDeDz = GetEmptyKahanAcc();

        //������� ��������� �������� �������� ����, �� ��������� �������� ������������
        for (
            int nextNeuronIndex = get_local_id(0),
                nextWeightIndex = ComputeWeightIndex(nextLayerNeuronCount, neuronIndex) + get_local_id(0);
            nextNeuronIndex < nextLayerNeuronCount; 
            nextNeuronIndex += get_local_size(0),
            nextWeightIndex += get_local_size(0)
            )
//        for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; nextNeuronIndex++, nextWeightIndex++)
        {
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


        float nOut = currentLayerNET[neuronIndex];
        currentDeDz *= <firstDerivative_nOut>;
        currentLayerDeDz[neuronIndex] = currentDeDz;

        //������� ��������� � ����� �������� �������� ���� �� ��������� �������� �����������
        for (
            int currentWeightIndex = get_local_id(0);
            currentWeightIndex < previousLayerNeuronCount; 
            currentWeightIndex += get_local_size(0)
            )
//        for (
//            int currentWeightIndex = 0; 
//            currentWeightIndex < previousLayerNeuronCount; 
//            ++currentWeightIndex)
        {
            float prevOut = previousLayerLastState[currentWeightIndex];

            float regularizationCoef = <nabla_regularization2>;
            float coef = prevOut + regularizationCoef;
            float n = learningRate * currentDeDz * coef;

            <nabla_update>
        }
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
    float dataCount)
{
    int neuronIndex = get_group_id(0);

    if(neuronIndex < currentLayerNeuronCount)
    {

        float nOut = currentLayerNET[neuronIndex];

        float n =
            <firstDerivative_nOut>
            * (desiredOutput[neuronIndex] - currentLayerLastState[neuronIndex]);  //!!! HalfSquaredEuclidianDistance, refactor!

        currentLayerDeDz[neuronIndex] = n;

        int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

        for (
            int weightIndex = get_local_id(0);
            weightIndex < previousLayerNeuronCountTotal; 
            weightIndex += get_local_size(0)
            )
//        for (
//            int weightIndex = 0; 
//            weightIndex < previousLayerNeuronCountTotal; 
//            ++weightIndex)
        {
            float deltaWeight =
                learningRate *
                n *
                (previousLayerLastState[weightIndex] + <weight_regularization2>);

            <weight_update>
        }
    }
}
";

        #endregion

        #region update weight kernel source

        public static string GetUpdateWeightKernelSource(int blockDim)
        {
            var ks = UpdateWeightKernelSource.Replace("{BLOCK_DIM}", blockDim.ToString());

            return
                ks;
        }

        private const string UpdateWeightKernelSource = @"
#define BLOCK_DIM {BLOCK_DIM}

__kernel void UpdateWeightAndTransposedWeightsKernel(
    __global float * currentLayerWeights,
    __global float * nabla,
    __global float * transposedCurrentLayerWeights,
    __local float* block,
    int currentLayerNeuronCountWithoutBias,
    int previousLayerNeuronCountWithBias, 
    float batchSize
    )
{
    // read the matrix tile into shared memory
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex < currentLayerNeuronCountWithoutBias) && (yIndex < previousLayerNeuronCountWithBias))
    {
        unsigned int index_in = yIndex * currentLayerNeuronCountWithoutBias + xIndex;

        float additive = nabla[index_in] / batchSize;
        float newValue = currentLayerWeights[index_in] + additive;

        block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = newValue;
        currentLayerWeights[index_in] = newValue;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // write the transposed matrix tile to global memory
    xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
    if((xIndex < previousLayerNeuronCountWithBias) && (yIndex < currentLayerNeuronCountWithoutBias))
    {
        unsigned int index_out = yIndex * previousLayerNeuronCountWithBias + xIndex;
        transposedCurrentLayerWeights[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
    }



/*
    int neuronIndex = get_global_id(0);
    int weightIndex = get_global_id(1);

    int i = neuronIndex * previousLayerNeuronCountWithBias + weightIndex;

    float additive = nabla[i] / batchSize;
    float newValue = currentLayerWeights[i] + additive;

    currentLayerWeights[i] = newValue;

    int transposedi = currentLayerNeuronCountWithoutBias * weightIndex + neuronIndex;
    transposedCurrentLayerWeights[transposedi] = newValue;
//*/
}
";

        #endregion

    }
}