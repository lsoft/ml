using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.NeuralNet.Train.Algo
{
    public class VOpenCLBackpropAlgorithm : BaseTrainAlgorithm
    {
        private readonly VNNCLProvider _clProvider;

        private readonly Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private readonly Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private readonly Kernel _updateWeightKernel;

        private readonly MemFloat[] _deDz;
        private readonly MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private readonly int _outputLength;

        public VOpenCLBackpropAlgorithm(
            MultiLayerNeuralNetwork network,
            ILearningAlgorithmConfig config,
            MultilayerTrainProcessDelegate validation,
            VNNCLProvider clProvider,
            int randomSeed)
            : base(network, config, validation)
        {
            _config = config;
            _clProvider = clProvider;
            _outputLength = _clProvider.Network.Layers.Last().NonBiasNeuronCount;
            _hiddenKernelIncrement = new Kernel[_clProvider.Network.Layers.Length];
            _hiddenKernelOverwrite = new Kernel[_clProvider.Network.Layers.Length];
            _outputKernelIncrement = new Kernel[_clProvider.Network.Layers.Length];
            _outputKernelOverwrite = new Kernel[_clProvider.Network.Layers.Length];

            for (var layerIndex = 1; layerIndex < _clProvider.Network.Layers.Length; layerIndex++)
            {
                _hiddenKernelIncrement[layerIndex] = _clProvider.CreateKernel(GetIncrementCalculationKernelsSource(layerIndex), "HiddenLayerTrain");
                _hiddenKernelOverwrite[layerIndex] = _clProvider.CreateKernel(GetOverwriteCalculationKernelsSource(layerIndex), "HiddenLayerTrain");

                _outputKernelIncrement[layerIndex] = _clProvider.CreateKernel(GetIncrementCalculationKernelsSource(layerIndex), "OutputLayerTrain");
                _outputKernelOverwrite[layerIndex] = _clProvider.CreateKernel(GetOverwriteCalculationKernelsSource(layerIndex), "OutputLayerTrain");
            }

            //определяем кернел обновления весов
            _updateWeightKernel = _clProvider.CreateKernel(_updateWeightKernelSource, "UpdateWeightKernel");

            _nablaWeights = new MemFloat[_network.Layers.Length];
            _deDz = new MemFloat[_network.Layers.Length];
        }

        protected override void PreTrainInit(DataSet data)
        {
            //создаем массивы смещений по весам и dedz
            for (var i = 1; i < _network.Layers.Length; i++)
            {
                var lastLayer = i == (_network.Layers.Length - 1);
                var biasNeuronCount = lastLayer ? 0 : 1;

                _nablaWeights[i] = _clProvider.CreateFloatMem(
                    (_network.Layers[i].Neurons.Length - biasNeuronCount) * _network.Layers[i].Neurons[0].Weights.Length,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

                _deDz[i] = _clProvider.CreateFloatMem(
                    _network.Layers[i].NonBiasNeuronCount,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            }

            //создаем объекты желаемых выходных данных (т.е. правильных ответов сети)
            _desiredOutput = _clProvider.CreateFloatMem(
                _outputLength,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);
        }

        protected override void TrainEpoche(DataSet data, string epocheRoot, float learningRate)
        {
            #region one epoche

            //переносим образ сети в объекты OpenCL
            _clProvider.Unpack();

            //гоним на устройство
            foreach (var nw in _nablaWeights)
            {
                if (nw != null)
                {
                    nw.Write(BlockModeEnum.NonBlocking);
                }
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //process data set
            var currentIndex = 0;
            do
            {
                //process one batch

                for (int batchIndex = currentIndex, inBatchIndex = 0; batchIndex < currentIndex + _config.BatchSize && batchIndex < data.Count; ++batchIndex, ++inBatchIndex)
                {
                    //train data
                    var trainData = data[batchIndex];

                    //прописываем значения во входные нейроны
                    _clProvider.Network.Layers[0].Compute(trainData.Input);

                    //записываем входные нейроны в объекты OpenCL
                    _clProvider.UnpackInput();

                    //---------------------------- forward pass ----------------------------

                    _network.ExecuteComputation();

                    //---------------------------- backward pass, error propagation ----------------------------

                    //отправляем на OpenCL желаемые выходы
                    trainData.Output.CopyTo(_desiredOutput.Array, 0);
                    _desiredOutput.Write(BlockModeEnum.Blocking);

                    //output layer
                    //скорее всего выгоднее посчитать на C#, так как на выходном слое мало нейронов
                    //и гонять данные туда и обратно на OpenCL устройство дороже, чем считать в C#
                    //но гонять данные все равно придется, так как скрытые слои работают на OpenCL
                    //поэтому получается выгодно считать и выходной слой на OpenCL (теряем на нем, но
                    //суммарно выигрыш)
                    var outputLayerIndex = _network.Layers.Length - 1;

                    var outputLayer = _network.Layers[outputLayerIndex];
                    var preOutputLayer = _network.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

                    if (inBatchIndex == 0)
                    {
                        _outputKernelOverwrite.Last()
                            .SetKernelArgMem(0, _clProvider.LastStateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(1, _clProvider.LastStateMem[outputLayerIndex])
                            .SetKernelArgMem(2, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(3, _desiredOutput)
                            .SetKernelArgMem(4, _clProvider.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(5, outputNablaLayer)
                            .SetKernelArg(6, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(9, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(10, 4, learningRate)
                            .SetKernelArg(11, 4, _config.RegularizationFactor)
                            .SetKernelArg(12, 4, (float)(data.Count))
                            .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                    }
                    else
                    {
                        _outputKernelIncrement.Last()
                            .SetKernelArgMem(0, _clProvider.LastStateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(1, _clProvider.LastStateMem[outputLayerIndex])
                            .SetKernelArgMem(2, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(3, _desiredOutput)
                            .SetKernelArgMem(4, _clProvider.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(5, outputNablaLayer)
                            .SetKernelArg(6, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(9, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(10, 4, learningRate)
                            .SetKernelArg(11, 4, _config.RegularizationFactor)
                            .SetKernelArg(12, 4, (float)(data.Count))
                            .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                    }

                    //hidden layers
                    //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                    //тут паралеллизовать нечего
                    for (int hiddenLayerIndex = _network.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                    {
                        //определяем слои
                        var prevLayer = _network.Layers[hiddenLayerIndex - 1];
                        var currentLayer = _network.Layers[hiddenLayerIndex];
                        var nextLayer = _network.Layers[hiddenLayerIndex + 1];

                        if (inBatchIndex == 0)
                        {
                            _hiddenKernelOverwrite[hiddenLayerIndex]
                                .SetKernelArgMem(0, _clProvider.LastStateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(1, _clProvider.LastStateMem[hiddenLayerIndex])
                                .SetKernelArgMem(2, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(4, _clProvider.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(5, _clProvider.WeightMem[hiddenLayerIndex + 1])
                                .SetKernelArgMem(6, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(7, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(10, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(11, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(12, 4, learningRate)
                                .SetKernelArg(13, 4, _config.RegularizationFactor)
                                .SetKernelArg(14, 4, (float)(data.Count))
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                        }
                        else
                        {
                            _hiddenKernelIncrement[hiddenLayerIndex]
                                .SetKernelArgMem(0, _clProvider.LastStateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(1, _clProvider.LastStateMem[hiddenLayerIndex])
                                .SetKernelArgMem(2, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(4, _clProvider.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(5, _clProvider.WeightMem[hiddenLayerIndex + 1])
                                .SetKernelArgMem(6, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(7, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(10, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(11, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(12, 4, learningRate)
                                .SetKernelArg(13, 4, _config.RegularizationFactor)
                                .SetKernelArg(14, 4, (float)(data.Count))
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                        }
                    }

                    //// Make sure we're done with everything that's been requested before
                    //_clProvider.QueueFinish();

                    int logStep = data.Count / 100;
                    if (logStep > 0 && currentIndex % logStep == 0)
                    {
                        Console.Write("Epoche progress: " + (currentIndex * 100 / data.Count) + "%, " + DateTime.Now.ToString() + "      ");
                        Console.SetCursorPosition(0, Console.CursorTop);
                    }
                }

                //update weights and bias into opencl memory wrappers

                for (int layerIndex = 1; layerIndex < _network.Layers.Length; ++layerIndex)
                {
                    var weightMem = _clProvider.WeightMem[layerIndex];
                    var nablaMem = _nablaWeights[layerIndex];

                    const int perKernelFloats = 1500; //по 1500 флоатов на кернел (должно быть кратно 4м!!!)

                    var kernelCount = weightMem.Array.Length / perKernelFloats;
                    if (weightMem.Array.Length % perKernelFloats > 0)
                    {
                        kernelCount++;
                    }

                    _updateWeightKernel
                        .SetKernelArgMem(0, weightMem)
                        .SetKernelArgMem(1, nablaMem)
                        .SetKernelArg(2, 4, weightMem.Array.Length)
                        .SetKernelArg(3, 4, perKernelFloats)
                        .SetKernelArg(4, 4, (float)(_config.BatchSize))
                        .EnqueueNDRangeKernel(kernelCount);
                }

                // Make sure we're done with everything that's been requested before
                _clProvider.QueueFinish();

                currentIndex += _config.BatchSize;
            } while (currentIndex < data.Count);

            #endregion

            //конец эпохи обучения

            //считываем веса с устройства
            foreach (var wm in _clProvider.WeightMem)
            {
                if (wm != null)
                {
                    wm.Read(BlockModeEnum.Blocking);
                }
            }


            //write new weights and biases into network
            for (int layerIndex = 1; layerIndex < _network.Layers.Length; ++layerIndex)
            {
                var layer = _network.Layers[layerIndex];
                var weightLayer = _clProvider.WeightMem[layerIndex];

                var weightShiftIndex = 0;
                for (int neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; ++neuronIndex)
                {
                    var neuron = layer.Neurons[neuronIndex];

                    var weightCount = neuron.Weights.Length;

                    Array.Copy(weightLayer.Array, weightShiftIndex, neuron.Weights, 0, weightCount);
                    weightShiftIndex += weightCount;
                }
            }
        }

        #region calculation kernels source

        private string GetOverwriteCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _network.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
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

        private string GetIncrementCalculationKernelsSource(int layerIndex)
        {
            var fDerivative = _network.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
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
    float currentDeDz = 0;
    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
    {
        int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex; //не векторизуется:(

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerDeDz[nextNeuronIndex];
        float multiplied = nextWeight * nextNabla;

        currentDeDz += multiplied;
    }

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

        private const string _updateWeightKernelSource = @"
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


        //!!! попробовать обновлять веса не многими воркерами, а одним (несколькими? по количеству слоев?) воркером в цикле
    }
}

