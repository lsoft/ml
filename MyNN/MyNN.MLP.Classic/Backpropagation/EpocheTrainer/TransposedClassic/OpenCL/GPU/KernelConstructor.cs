using System;
using System.Linq;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU
{
    /// <summary>
    /// Kernel source provider for classic backpropagation epoche trainer with transposed weights that enables CPU-OpenCL
    /// </summary>
    public class KernelConstructor
    {
        private const string DerivativeMethodName = "Derivative";
        private const string MetricMethodName = "CalculateMetric";

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
        currentDeDz *= <DerivativeMethodCall>(nOut);
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
        float deri = <DerivativeMethodCall>(nOut);

        float metric = <MetricMethodCall>(
            currentLayerLastState,
            desiredOutput,
            neuronIndex
            );

        float n = deri * metric;

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

        public const string UpdateWeightKernelSource = @"
__kernel void UpdateWeightKernel(
    __global float * currentLayerWeights,
    const __global float * nabla,
    const float batchSize,
    const int totalCount
    )
{
    int gi = get_global_id(0);

    float shift = nabla[gi] / batchSize;
    currentLayerWeights[gi] += shift;
}
";

        #endregion

    }
}