using System;
using System.Linq;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU.KernelText
{
    /// <summary>
    /// Kernel source provider for dropout backpropagation epoche trainer that enables GPU-OpenCL
    /// </summary>
    internal class KernelTextProviderWithRegularization : IKernelTextProvider
    {
        private const string DerivativeMethodName = "Derivative";
        private const string MetricMethodName = "CalculateMetric";

        private readonly ILearningAlgorithmConfig _config;

        public KernelTextProviderWithRegularization(
            ILearningAlgorithmConfig config
            )
        {
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (Math.Abs(config.RegularizationFactor) < float.Epsilon)
            {
                throw new ArgumentException("Math.Abs(config.RegularizationFactor) < float.Epsilon");
            }

            _config = config;
        }

        #region calculation kernels source

        public string GetOverwriteCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }

            var fDerivative = layerConfiguration.LayerActivationFunction.GetOpenCLDerivativeMethod(DerivativeMethodName, VectorizationSizeEnum.NoVectorization);
            var result = CalculationKernelsSource.Replace("<DerivativeMethodBody>", fDerivative);

            result = result.Replace(
                "<MetricMethodBody>",
                _config.TargetMetrics.GetOpenCLPartialDerivative(
                    MetricMethodName,
                    VectorizationSizeEnum.NoVectorization,
                    MemModifierEnum.Global,
                    layerConfiguration.TotalNeuronCount
                    )
                );

            result = result.Replace(
                "<MetricMethodCall>",
                MetricMethodName
                );

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] = n;
");

            result =
                result.Replace("<weight_update>",@"
        nabla[nablaNeuronShift + weightIndex] = deltaWeight;
");

            result =
                result.Replace("<bias_update>", @"
        nablaBias[neuronIndex] = deltaBias;
");

            return result;
        }

        public string GetIncrementCalculationKernelsSource(
            ILayerConfiguration layerConfiguration
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }

            var fDerivative = layerConfiguration.LayerActivationFunction.GetOpenCLDerivativeMethod(DerivativeMethodName, VectorizationSizeEnum.NoVectorization);
            var result = CalculationKernelsSource.Replace("<DerivativeMethodBody>", fDerivative);

            result = result.Replace(
                "<MetricMethodBody>",
                _config.TargetMetrics.GetOpenCLPartialDerivative(
                    MetricMethodName,
                    VectorizationSizeEnum.NoVectorization,
                    MemModifierEnum.Global,
                    layerConfiguration.TotalNeuronCount
                    )
                );

            result = result.Replace(
                "<MetricMethodCall>",
                MetricMethodName
                );

            result = result.Replace("<DerivativeMethodCall>", DerivativeMethodName);

            result =
                result.Replace("<nabla_update>", @"
        nabla[currentNablaIndex + currentWeightIndex] += n;
");

            result =
                result.Replace("<weight_update>", @"
        nabla[nablaNeuronShift + weightIndex] += deltaWeight;
");

            result =
                result.Replace("<bias_update>", @"
        nablaBias[neuronIndex] += deltaBias;
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
    __global float * currentLayerDeDz,
    __global float * currentLayerWeights,
            
    __global float * nabla,

    const __global uint * masks,
    int maskShift,
    uint bitmask,

    int previousLayerNeuronCount,
    int currentLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount,

    __local float * local_accum,

    __global float * dedys,

    __global float* currentLayerBias,
    __global float* nablaBias
    )
{
    int neuronIndex = get_group_id(0);

    if(neuronIndex < currentLayerNeuronCount)
    {
        int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

        //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего уже выполнен
        float dedy = dedys[neuronIndex];

        //вычисляем маску
        uint maski = masks[neuronIndex + maskShift];
        float mask = ((maski & bitmask) > 0) ? (float)1 : (float)0;

        float nOut = currentLayerNET[neuronIndex];
        float currentDeDz = dedy * <DerivativeMethodCall>(nOut);
        currentDeDz *= mask;
        currentLayerDeDz[neuronIndex] = currentDeDz;

        for (
            int currentWeightIndex = get_local_id(0);
            currentWeightIndex < previousLayerNeuronCount; 
            currentWeightIndex += get_local_size(0)
            )
        {
            float prevOut = previousLayerLastState[currentWeightIndex];

            float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
            float coef = prevOut + regularizationCoef;
            float n = learningRate * currentDeDz * coef;

            n *= mask; //регуляризация может изменить нуль, который был в n благодаря умножению на currentDeDz

            <nabla_update>
        }

        if(get_local_id(0) == 0)
        {
            float deltaBias =
                learningRate *
                currentDeDz *
                (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

            <bias_update>
        }
        barrier(CLK_LOCAL_MEM_FENCE);

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
    float dataCount,

    __global float* currentLayerBias,
    __global float* nablaBias
    )
{
    int neuronIndex = get_group_id(0);

    if(neuronIndex < currentLayerNeuronCount)
    {
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
        //int nablaNeuronShift4 = nablaNeuronShift / 4;
        //int nablaNeuronShift4Shift = nablaNeuronShift - nablaNeuronShift4 * 4;

        //невекторизованная часть
        for (
            int weightIndex = get_local_id(0);
            weightIndex < previousLayerNeuronCountTotal; 
            weightIndex += get_local_size(0)
            )
        {
            float deltaWeight =
                learningRate *
                n *
                (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

            <weight_update>
        }

        if(get_local_id(0) == 0)
        {
            float deltaBias =
                learningRate *
                n *
                (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

            <bias_update>
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }
}
";

        #endregion

        #region update weight kernel source

        public string UpdateWeightKernelSource
        {
            get
            {
                throw new NotSupportedException("Реализовано в кернел провайдере более высокого уровня");
            }
        }

        #endregion

    }
}