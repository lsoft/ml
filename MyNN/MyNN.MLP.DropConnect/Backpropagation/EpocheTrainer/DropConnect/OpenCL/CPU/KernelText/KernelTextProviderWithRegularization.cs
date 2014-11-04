using System;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.CPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables CPU-OpenCL
    /// </summary>
    internal class KernelTextProviderWithRegularization : IKernelTextProvider
    {
        private const string DerivativeMethodName = "Derivative";

        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        public KernelTextProviderWithRegularization(
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
            if (Math.Abs(config.RegularizationFactor) < float.Epsilon)
            {
                throw new ArgumentException("Math.Abs(config.RegularizationFactor) < float.Epsilon");
            }

            _mlp = mlp;
            _config = config;
        }

        #region calculation kernels source

        public string GetOverwriteCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLDerivativeMethod(DerivativeMethodName, VectorizationSizeEnum.NoVectorization);
            var result = CalculationKernelsSource.Replace("<DerivativeMethodBody>", fDerivative);

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] = n;
");

            result =
                result.Replace("<weight_update>",@"
        nabla[nablaNeuronShift + weightIndex] = deltaWeight;
");

            return result;
        }

        public string GetIncrementCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLDerivativeMethod(DerivativeMethodName, VectorizationSizeEnum.NoVectorization);
            var result = CalculationKernelsSource.Replace("<DerivativeMethodBody>", fDerivative);

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] += n;
");

            result =
                result.Replace("<weight_update>", @"
        nabla[nablaNeuronShift + weightIndex] += deltaWeight;
");

            return result;
        }


        private const string CalculationKernelsSource = @"
<DerivativeMethodBody>

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

    __global uint * mask,

    int previousLayerNeuronCount,
    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount,

    uint bitmask
    )
{
    int neuronIndex = get_global_id(0);

    int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);


    //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)

    KahanAccumulator accDeDz = GetEmptyKahanAcc();
    for (
        int nextNeuronIndex = 0;
        nextNeuronIndex < nextLayerNeuronCount; 
        nextNeuronIndex++
        )
    {
        int nextWeightIndex = 
            ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + 
            neuronIndex;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        KahanAddElement(&accDeDz, multiplied);
    }

    float currentDeDz = accDeDz.Sum;


    float nOut = currentLayerNET[neuronIndex];
    currentDeDz *= <DerivativeMethodCall>(nOut);
    currentLayerDeDz[neuronIndex] = currentDeDz;

    //невекторизованная часть
    for (
        int currentWeightIndex = 0; 
        currentWeightIndex < previousLayerNeuronCount; 
        ++currentWeightIndex)
    {
        float prevOut = previousLayerLastState[currentWeightIndex];

        float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
        float coef = prevOut + regularizationCoef;

        uint mask1i = mask[currentNablaIndex + currentWeightIndex];
        float mask1 = ((mask1i & bitmask) > 0) ? (float)1 : (float)0;

        float n = learningRate * currentDeDz * mask1 * coef;

        <nabla_update>
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

    __global uint * mask,

    int previousLayerNeuronCountTotal,
    int currentLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount,

    uint bitmask
    )
{
    int neuronIndex = get_global_id(0);

    float nOut = currentLayerNET[neuronIndex];
    float deri = <DerivativeMethodCall>(nOut);

    float n =
        deri
        * (desiredOutput[neuronIndex] - currentLayerLastState[neuronIndex]); //!!! HalfSquaredEuclidianDistance, refactor!

    currentLayerDeDz[neuronIndex] = n;

    int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    int nablaNeuronShift4 = nablaNeuronShift / 4;
    int nablaNeuronShift4Shift = nablaNeuronShift - nablaNeuronShift4 * 4;

    //невекторизованная часть
    for (
        int weightIndex = 0; 
        weightIndex < previousLayerNeuronCountTotal; 
        ++weightIndex)
    {
        uint mask1i = mask[nablaNeuronShift + weightIndex];
        float mask1 = ((mask1i & bitmask) > 0) ? (float)1 : (float)0;

        float deltaWeight =
            learningRate *
            n *
            mask1 *
            (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

        <weight_update>
    }
}
";

        #endregion

        #region update weight kernel source

        public string UpdateWeightKernelSource
        {
            get
            {
                return @"
__kernel void UpdateWeightKernel(
    __global float * currentLayerWeights,
    __global float * nabla,
    int count, //общее количество флоатов для обработки (для всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount, //количество флоатов для обработки ОДНИМ кернелом (должно быть кратно 4м!!!)
    float batchSize)
{
    int kernelIndex = get_global_id(0);
    
    int d1StartIndex = kernelIndex * kernelDataCount;
    int d1Count = min(kernelDataCount, count - d1StartIndex);

    int d4StartIndex = d1StartIndex / 4;
    int d4Count = d1Count / 4;
    
    int d1StartRemainder = d1StartIndex + d4Count * 4;

    for(int cc = d4StartIndex; cc < d4StartIndex + d4Count; cc++)
    {
        float4 currentLayerWeights4 = vload4(cc, currentLayerWeights);
        float4 nabla4 = vload4(cc, nabla);

        float4 result = currentLayerWeights4 + nabla4 / batchSize;

        vstore4(
            result,
            cc,
            currentLayerWeights);
    }

    for(int cc = d1StartRemainder; cc < d1StartIndex + d1Count; cc++)
    {
        currentLayerWeights[cc] += nabla[cc] / batchSize;
    }
}
";
            }
        }

        #endregion

    }
}