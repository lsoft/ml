using System;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCLTranspose
{
    public class TransposeKernelConstructor
    {
        private readonly MLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        public TransposeKernelConstructor(
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

            result = result.Replace("<vectorized_nabla_update>", string.Empty);

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


            result = result.Replace("<vectorized_weight_update>", string.Empty);

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
            var result = _calculationKernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

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
                result.Replace("<vectorized_weight_update>", @"
        float4 nabla4 = vload4(nablaNeuronShift4 + weightIndex4, nabla + nablaNeuronShift4Shift);
        deltaWeight4 += nabla4;
");

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


        private const string _calculationKernelsSource = @"
int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

//const __constant float _alpha = 0.2;
//const __constant float _beta = 1.0;

__kernel void HiddenLayerTrain(
    __global float * previousLayerLastState,
    __global float * currentLayerLastState,
    __global float * currentLayerDeDz,
    __global float * nextLayerDeDz,

    __global float * currentLayerWeights,
    __global float * nextLayerWeights,
            
    __global float * nabla,

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

    int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

    //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего
    float16 currentDeDz16 = 0;
    
    int nextNeuronIndex = 0;

    int nextWeightIndex = ComputeWeightIndex(nextLayerNeuronCount, neuronIndex);
    int nextWeightShift = nextWeightIndex % 16;
    nextWeightIndex = (nextWeightIndex - nextWeightShift) / 16;

    for (; nextNeuronIndex < (nextLayerNeuronCount / 16); nextNeuronIndex++, nextWeightIndex ++)
    {
        float16 nextWeight16 = vload16(nextWeightIndex, nextLayerWeights + nextWeightShift);
        float16 nextNabla16 = vload16(nextNeuronIndex, nextLayerDeDz);
        float16 multiplied16 = nextWeight16 * nextNabla16;

        currentDeDz16 += multiplied16;
    }

    nextNeuronIndex = nextNeuronIndex * 16;
    nextWeightIndex = nextWeightIndex * 16 + nextWeightShift;

    float currentDeDz = 
          currentDeDz16.s0 
        + currentDeDz16.s1 
        + currentDeDz16.s2 
        + currentDeDz16.s3
        + currentDeDz16.s4
        + currentDeDz16.s5
        + currentDeDz16.s6
        + currentDeDz16.s7
        + currentDeDz16.s8
        + currentDeDz16.s9
        + currentDeDz16.sa
        + currentDeDz16.sb
        + currentDeDz16.sc
        + currentDeDz16.sd
        + currentDeDz16.se
        + currentDeDz16.sf
        ;

    for (; nextNeuronIndex < nextLayerNeuronCount; nextNeuronIndex += 1, nextWeightIndex += 1)
    {
        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        currentDeDz += multiplied;
    }


//    //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (НЕ ВЕКТОРИЗОВАНО)
//    float currentDeDz = 0;
//    
//    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
//    {
//        int nextWeightIndex = ComputeWeightIndex(nextLayerNeuronCount, neuronIndex) + nextNeuronIndex;
//
//        float nextWeight = nextLayerWeights[nextWeightIndex];
//        float nextNabla = nextLayerDeDz[nextNeuronIndex];
//        float multiplied = nextWeight * nextNabla;
//
//        currentDeDz += multiplied;
//    }

    float nOut = currentLayerLastState[neuronIndex];
    currentDeDz *= <firstDerivative_nOut>;//nOut * (1 - nOut);
    currentLayerDeDz[neuronIndex] = currentDeDz;

    int currentNablaIndex4 = currentNablaIndex / 4;
    int currentNablaIndex4Shift = currentNablaIndex - currentNablaIndex4 * 4;

    //просчет изменений в весах нейронов текущего слоя по состоянию нейронов предыдущего
    //векторизованная часть
    for (
        int currentWeightIndex4 = 0; 
        currentWeightIndex4 < previousLayerNeuronCount4; 
        ++currentWeightIndex4)
    {
        float4 prevOut = vload4(currentWeightIndex4, previousLayerLastState);

        float4 regularizationCoef = <nabla_regularization1>;
        float4 coef = prevOut + regularizationCoef;
        float4 n = learningRate * currentDeDz * coef;

        <vectorized_nabla_update>

        vstore4(
            n,
            currentNablaIndex4 + currentWeightIndex4,
            nabla + currentNablaIndex4Shift);
    }

    //невекторизованная часть (добиваем остатки)
    for (
        int currentWeightIndex = previousLayerNeuronCount4M4; 
        currentWeightIndex < previousLayerNeuronCount; 
        ++currentWeightIndex)
    {
        float prevOut = previousLayerLastState[currentWeightIndex];

        float regularizationCoef = <nabla_regularization2>;
        float coef = prevOut + regularizationCoef;
        float n = learningRate * currentDeDz * coef;

        <nabla_update>
    }
}

__kernel void OutputLayerTrain(
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

    float nOut = currentLayerLastState[neuronIndex];

    float n =
        <firstDerivative_nOut> //rOut * (1 - nOut)
        * (desiredOutput[neuronIndex] - nOut);

    currentLayerDeDz[neuronIndex] = n;

    int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    int nablaNeuronShift4 = nablaNeuronShift / 4;
    int nablaNeuronShift4Shift = nablaNeuronShift - nablaNeuronShift4 * 4;

    //векторизованная часть
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
            (previousLayerLastState4 + <weight_regularization1>);

        <vectorized_weight_update>

        vstore4(
            deltaWeight4,
            nablaNeuronShift4 + weightIndex4,
            nabla + nablaNeuronShift4Shift);
    }

    //добираем остатки (от нуля до 3 флоатов)
    for (
        int weightIndex = previousLayerNeuronCount4M4; 
        weightIndex < previousLayerNeuronCountTotal; 
        ++weightIndex)
    {
        float deltaWeight =
            learningRate *
            n *
            (previousLayerLastState[weightIndex] + <weight_regularization2>);

        <weight_update>
    }
}
";

        #endregion

        #region update weight kernel source

        public const string UpdateWeightKernelSource = @"
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

        #endregion

    }
}