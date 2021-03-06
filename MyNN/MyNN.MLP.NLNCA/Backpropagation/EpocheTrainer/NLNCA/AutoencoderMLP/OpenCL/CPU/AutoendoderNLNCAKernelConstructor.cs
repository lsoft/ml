using System;
using System.Linq;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.AutoencoderMLP.OpenCL.CPU
{
    /// <summary>
    /// Kernel source provider for regularized NLNCA backpropagation epoche trainer for autoencoders that enables CPU-OpenCL.
    /// </summary>
    public class AutoendoderNLNCAKernelConstructor
    {
        private const string DerivativeMethodName = "Derivative";
        private const string MetricMethodName = "CalculateMetric";

        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        public AutoendoderNLNCAKernelConstructor(
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
                    _mlp.Layers.Last().TotalNeuronCount
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

            result =
                result.Replace(
                    "<nabla_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
����� ������ � �������������!!! ������ ������ ��� CPU (������ ���� ����� �� currentLayerWeights[currentWeightIndex4] , �  �� 4; ���� ������������ vload4)
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

            result =
                result.Replace("<bias_update>", @"
        nablaBias[neuronIndex] = deltaBias;
");

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
                    _mlp.Layers.Last().TotalNeuronCount
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
                result.Replace(
                    "<nabla_regularization1>",
                    (Math.Abs(_config.RegularizationFactor) < float.Epsilon
                        ? "0"
                        : @"
����� ������ � �������������!!! ������ ������ ��� CPU (������ ���� ����� �� currentLayerWeights[currentWeightIndex4] , �  �� 4; ���� ������������ vload4)
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
    __global float * currentLayerLastState,
    __global float * currentLayerDeDz,
    __global float * nextLayerDeDz,

    __global float * currentLayerWeights,
    __global float * nextLayerWeights,
            
    __global float * nabla,

    __global float * dodf,

    int previousLayerNeuronCount4,
    int previousLayerNeuronCount4M4,
    int previousLayerNeuronCount,
    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    int takeIntoAccount,
    float lambda,

    float learningRate,
    float regularizationFactor,
    float dataCount,

    __global float* currentLayerBias,
    __global float* nablaBias
    )
{
    int neuronIndex = get_global_id(0);

    int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

    //������� ��������� �������� �������� ����, �� ��������� �������� ������������
    float currentDeDz = 0;
    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
    {
        int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount, nextNeuronIndex) + neuronIndex; //�� �������������:(

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        currentDeDz += multiplied;
    }

    //���� �������������� ������ ������ � ����� ���,
    //�� ������� ����������� �������� NCA, 
    //�� ��������� �������� NCA
    if (neuronIndex < takeIntoAccount)
    {
        currentDeDz += lambda * dodf[neuronIndex];
    }

    float nOut = currentLayerNET[neuronIndex];
    currentDeDz *= <DerivativeMethodCall>(nOut);
    currentLayerDeDz[neuronIndex] = currentDeDz;

    int currentNablaIndex4 = currentNablaIndex / 4;
    int currentNablaIndex4Shift = currentNablaIndex - currentNablaIndex4 * 4;

    //������� ��������� � ����� �������� �������� ���� �� ��������� �������� �����������
    //��������������� �����
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

    //����������������� ����� (�������� �������)
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

    float deltaBias =
        learningRate *
        currentDeDz *
        1;//(1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

    <bias_update>

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

    float lambda,

    float learningRate,
    float regularizationFactor,
    float dataCount,

    __global float* currentLayerBias,
    __global float* nablaBias
    )
{
    int neuronIndex = get_global_id(0);

    float nOut = currentLayerNET[neuronIndex];
    float deri = <DerivativeMethodCall>(nOut);

    float metric = <MetricMethodCall>(
        currentLayerLastState,
        desiredOutput,
        neuronIndex
        );

    float n = deri * lambda * metric;

    currentLayerDeDz[neuronIndex] = n;

    int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    int nablaNeuronShift4 = nablaNeuronShift / 4;
    int nablaNeuronShift4Shift = nablaNeuronShift - nablaNeuronShift4 * 4;

    //��������������� �����
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

    //�������� ������� (�� ���� �� 3 �������)
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

    float deltaBias =
        learningRate *
        n *
        1;//(1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

    <bias_update>

}
";

        #endregion

        #region update weight kernel source

        public const string UpdateWeightKernelSource = @"
__kernel void UpdateWeightKernel(
    __global float * currentLayerWeights,
    __global float * nablaWeights,
    int weightCount, //����� ���������� ������� ��� ��������� (��� ���� ��������, ����� currentLayerWeights, ����� nabla)
    int kernelDataCount, //���������� ������� ��� ��������� ����� �������� (������ ���� ������ 4�!!!)
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

        #endregion

    }
}