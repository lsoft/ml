using System;
using System.Linq;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.CPU.KernelText
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer that enables CPU-OpenCL
    /// </summary>
    internal class KernelTextProviderWithRegularization : IKernelTextProvider
    {
        private const string DerivativeMethodName = "Derivative";
        private const string MetricMethodName = "CalculateMetric";

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

            result = result.Replace(
                "<MetricMethodBody>",
                _config.TargetMetrics.GetOpenCLPartialDerivative(
                    MetricMethodName,
                    VectorizationSizeEnum.NoVectorization,
                    MemModifierEnum.Global,
                    _mlp.Layers.Last().NonBiasNeuronCount
                    )
                );

            result = result.Replace(
                "<MetricMethodCall>",
                MetricMethodName
                );

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result = result.Replace("<vectorized_nabla_update>", string.Empty);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] = n;
");

            result = result.Replace("<vectorized_weight_update>", string.Empty);

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

            result = result.Replace(
                "<MetricMethodBody>",
                _config.TargetMetrics.GetOpenCLPartialDerivative(
                    MetricMethodName,
                    VectorizationSizeEnum.NoVectorization,
                    MemModifierEnum.Global,
                    _mlp.Layers.Last().NonBiasNeuronCount
                    )
                );

            result = result.Replace(
                "<MetricMethodCall>",
                MetricMethodName
                );

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result =
                result.Replace("<vectorized_nabla_update>", @"
        float4 nabla4 = vload4(currentNablaIndex4 + currentWeightIndex4, nabla + currentNablaIndex4Shift);
        n += nabla4;
");

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] += n;
");

            result =
                result.Replace("<vectorized_weight_update>", @"
        float4 nabla4 = vload4(nablaNeuronShift4 + weightIndex4, nabla + nablaNeuronShift4Shift);
        deltaWeight4 += nabla4;
");

            result =
                result.Replace("<weight_update>", @"
        nabla[nablaNeuronShift + weightIndex] += deltaWeight;
");

            return result;
        }


        private const string CalculationKernelsSource = @"
<DerivativeMethodBody>

<MetricMethodBody>

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

    const __global uint * masks,
    int maskShift,
    uint bitmask,

    int previousLayerNeuronCount4,
    int previousLayerNeuronCount4M4,
    int previousLayerNeuronCount,
    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount)
{
    int neuronIndex = get_global_id(0);

    //вычисл€ем маску
    uint maski = masks[neuronIndex + maskShift];
    float mask = ((maski & bitmask) > 0) ? (float)1 : (float)0;

    int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);


    //просчет состо€ни€ нейронов текущего сло€, по состо€нию нейронов последующего (with Kahan Algorithm)
    KahanAccumulator accDeDz = GetEmptyKahanAcc();
    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
    {
        int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex; //не векторизуетс€:(

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        KahanAddElement(&accDeDz, multiplied);
    }

    float currentDeDz = accDeDz.Sum;


    float nOut = currentLayerNET[neuronIndex];
    currentDeDz *= <DerivativeMethodCall>(nOut);
    currentDeDz *= mask;
    currentLayerDeDz[neuronIndex] = currentDeDz;

    int currentNablaIndex4 = currentNablaIndex / 4;
    int currentNablaIndex4Shift = currentNablaIndex - currentNablaIndex4 * 4;

    //просчет изменений в весах нейронов текущего сло€ по состо€нию нейронов предыдущего
    //векторизованна€ часть
    for (
        int currentWeightIndex4 = 0; 
        currentWeightIndex4 < previousLayerNeuronCount4; 
        ++currentWeightIndex4)
    {
        float4 prevOut = vload4(currentWeightIndex4, previousLayerLastState);
        float4 currentLayerWeights4 = vload4(currentWeightIndex4, currentLayerWeights);

        float4 regularizationCoef = regularizationFactor * currentLayerWeights4 / dataCount;
        float4 coef = prevOut + regularizationCoef;
        float4 n = learningRate * currentDeDz * coef;

        <vectorized_nabla_update>

        n *= mask; //регул€ризаци€ может изменить нуль, который был в n благодар€ умножению на currentDeDz

        vstore4(
            n,
            currentNablaIndex4 + currentWeightIndex4,
            nabla + currentNablaIndex4Shift);
    }

    //невекторизованна€ часть (добиваем остатки)
    for (
        int currentWeightIndex = previousLayerNeuronCount4M4; 
        currentWeightIndex < previousLayerNeuronCount; 
        ++currentWeightIndex)
    {
        float prevOut = previousLayerLastState[currentWeightIndex];

        float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
        float coef = prevOut + regularizationCoef;
        float n = learningRate * currentDeDz * coef;

        n *= mask; //регул€ризаци€ может изменить нуль, который был в n благодар€ умножению на currentDeDz

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

    int previousLayerNeuronCount4,
    int previousLayerNeuronCount4M4,
    int previousLayerNeuronCountTotal,
    int currentLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount)
{
    int neuronIndex = get_global_id(0);

    float nOut = currentLayerNET[neuronIndex];
    float deri = <DerivativeMethodCall>(nOut);

    float metric = <MetricMethodCall>(
        currentLayerLastState,
        desiredOutput,
        neuronIndex
        );

    float n = deri * metric;

    currentLayerDeDz[neuronIndex] = n;

    int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    int nablaNeuronShift4 = nablaNeuronShift / 4;
    int nablaNeuronShift4Shift = nablaNeuronShift - nablaNeuronShift4 * 4;

    //векторизованна€ часть
    for (
        int weightIndex4 = 0;
        weightIndex4 < previousLayerNeuronCount4;
        ++weightIndex4)
    {
        float4 previousLayerLastState4 = vload4(weightIndex4, previousLayerLastState);
        float4 currentLayerWeights4 = vload4(nablaNeuronShift4 + weightIndex4, currentLayerWeights + nablaNeuronShift4Shift);
        
        float4 deltaWeight4 = 
            learningRate *
            n *
            (previousLayerLastState4 + regularizationFactor * currentLayerWeights4 / dataCount);

        <vectorized_weight_update>

        vstore4(
            deltaWeight4,
            nablaNeuronShift4 + weightIndex4,
            nabla + nablaNeuronShift4Shift);
    }

    //добираем остатки (от нул€ до 3 флоатов)
    for (
        int weightIndex = previousLayerNeuronCount4M4; 
        weightIndex < previousLayerNeuronCountTotal; 
        ++weightIndex)
    {
        float deltaWeight =
            learningRate *
            n *
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
    int count, //общее количество флоатов дл€ обработки (дл€ всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount, //количество флоатов дл€ обработки ќƒЌ»ћ кернелом (должно быть кратно 4м!!!)
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