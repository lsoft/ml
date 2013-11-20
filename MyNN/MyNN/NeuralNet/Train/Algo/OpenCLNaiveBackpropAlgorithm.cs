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
    public class OpenCLNaiveBackpropAlgorithm : BaseTrainAlgorithm
    {
        private readonly MNNCLProvider _clProvider;
        private readonly Kernel[] _hiddenKernel;
        private readonly Kernel[] _outputKernel;
        private readonly Kernel _nablaKernel;
        private readonly Kernel _updateWeightKernel;

        private readonly MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private readonly int _outputLength;

        public OpenCLNaiveBackpropAlgorithm(
            MultiLayerNeuralNetwork network,
            ILearningAlgorithmConfig config,
            MultilayerTrainProcessDelegate validation,
            MNNCLProvider clProvider)
            : base(network, config, validation)
        {
            _config = config;
            _clProvider = clProvider;
            _outputLength = _clProvider.Network.Layers.Last().NonBiasNeuronCount;
            _hiddenKernel = new Kernel[_clProvider.Network.Layers.Length];
            _outputKernel = new Kernel[_clProvider.Network.Layers.Length];

            for (var layerIndex = 1; layerIndex < _clProvider.Network.Layers.Length; layerIndex++)
            {
                var fDerivative = _network.Layers[layerIndex].LayerActivationFunction.GetOpenCLFirstDerivative("nOut");
                var kernelsSource = _kernelsSource.Replace("<firstDerivative_nOut>", fDerivative);

                _hiddenKernel[layerIndex] = _clProvider.CreateKernel(kernelsSource, "HiddenLayerTrain");
                _outputKernel[layerIndex] = _clProvider.CreateKernel(kernelsSource, "OutputLayerTrain");
            }

            _nablaKernel = _clProvider.CreateKernel(_additionalKernelsSource, "ClearNabla");
            _updateWeightKernel = _clProvider.CreateKernel(_additionalKernelsSource, "UpdateWeightKernel");

            _nablaWeights = new MemFloat[_network.Layers.Length];
        }

        protected override void PreTrainInit(DataSet data)
        {
            //создаем массивы смещений по весам
            for (int i = 1; i < _network.Layers.Length; i++)
            {
                var lastLayer = i == (_network.Layers.Length - 1);
                var biasNeuronCount = lastLayer ? 0 : 1;

                _nablaWeights[i] = _clProvider.CreateFloatMem(
                    (_network.Layers[i].Neurons.Length - biasNeuronCount)*_network.Layers[i].Neurons[0].Weights.Length,
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
                #region clean accumulated error for batch, for weights and biases

                foreach (var nw in _nablaWeights)
                {
                    if (nw != null)
                    {
                        const int perKernelFloats = 1500; //по 1500 флоатов на кернел (должно быть кратно 4м!!!)

                        var kernelCount = nw.Array.Length / perKernelFloats;
                        if (nw.Array.Length % perKernelFloats > 0)
                        {
                            kernelCount++;
                        }

                        _nablaKernel
                            .SetKernelArgMem(0, nw)
                            .SetKernelArg(1, 4, nw.Array.Length)
                            .SetKernelArg(2, 4, perKernelFloats)
                            .EnqueueNDRangeKernel(kernelCount);
                    }
                }

                #endregion

                //process one batch

                for (var batchIndex = currentIndex; batchIndex < currentIndex + _config.BatchSize && batchIndex < data.Count; ++batchIndex)
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

                    _outputKernel.Last()
                        .SetKernelArgMem(0, _clProvider.NeuronMem[outputLayerIndex - 1])
                        .SetKernelArgMem(1, _clProvider.NeuronMem[outputLayerIndex])
                        .SetKernelArgMem(2, _desiredOutput)
                        .SetKernelArgMem(3, _clProvider.WeightMem[outputLayerIndex])
                        .SetKernelArgMem(4, outputNablaLayer)
                        .SetKernelArg(5, 4, preOutputLayer.Neurons.Length)
                        .SetKernelArg(6, 4, outputLayer.NonBiasNeuronCount)
                        .SetKernelArg(7, 4, learningRate)
                        .SetKernelArg(8, 4, _config.RegularizationFactor)
                        .SetKernelArg(9, 4, (float) (data.Count))
                        .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);

                    //hidden layers
                    //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                    //тут паралеллизовать нечего
                    for (int hiddenLayerIndex = _network.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                    {
                        //определяем слои
                        var prevLayer = _network.Layers[hiddenLayerIndex - 1];
                        var currentLayer = _network.Layers[hiddenLayerIndex];
                        var nextLayer = _network.Layers[hiddenLayerIndex + 1];


                        _hiddenKernel[hiddenLayerIndex]
                            .SetKernelArgMem(0, _clProvider.NeuronMem[hiddenLayerIndex - 1])
                            .SetKernelArgMem(1, _clProvider.NeuronMem[hiddenLayerIndex])
                            .SetKernelArgMem(2, _clProvider.NeuronMem[hiddenLayerIndex + 1])
                            .SetKernelArgMem(3, _clProvider.WeightMem[hiddenLayerIndex])
                            .SetKernelArgMem(4, _clProvider.WeightMem[hiddenLayerIndex + 1])
                            .SetKernelArgMem(5, _nablaWeights[hiddenLayerIndex])
                            .SetKernelArg(6, 4, prevLayer.Neurons.Length)
                            .SetKernelArg(7, 4, currentLayer.NonBiasNeuronCount)
                            .SetKernelArg(8, 4, nextLayer.NonBiasNeuronCount)
                            .SetKernelArg(9, 4, learningRate)
                            .SetKernelArg(10, 4, _config.RegularizationFactor)
                            .SetKernelArg(11, 4, (float) (data.Count))
                            .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                    }

                    //// Make sure we're done with everything that's been requested before
                    //_clProvider.QueueFinish();

                    int logStep = data.Count/100;
                    if (logStep > 0 && currentIndex % logStep == 0)
                    {
                        Console.Write("Epoche progress: " + (currentIndex*100/data.Count) + "%, " + DateTime.Now.ToString() + "      ");
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


        private string _kernelsSource = @"
typedef struct
{
    float LastNET;
    float LastState;
    float Dedz;
} Neuron;

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
    __global Neuron * previousLayerNeurons,
    __global Neuron * currentLayerNeurons,
    __global Neuron * nextLayerNeurons,

    __global float * currentLayerWeights,
    __global float * nextLayerWeights,
            
    __global float * nabla,

    int previousLayerNeuronCount,
    int currentLayerNeuronCount,
    int nextLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount)
{
    int neuronIndex = get_global_id(0);

    //currentLayerNeurons[neuronIndex].Dedz = 0.0;

    int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

    //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего
    float currentDeDz = 0;
    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
    {
        int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex;

        float nextWeight = nextLayerWeights[nextWeightIndex];
        float nextNabla = nextLayerNeurons[nextNeuronIndex].Dedz;
        float multiplied = nextWeight * nextNabla;

        //currentLayerNeurons[neuronIndex].Dedz += multiplied;
        currentDeDz += multiplied;
    }

    float nOut = currentLayerNeurons[neuronIndex].LastState;
    //currentLayerNeurons[neuronIndex].Dedz *= <firstDerivative_nOut>;//nOut * (1 - nOut);
    currentDeDz *= <firstDerivative_nOut>;//nOut * (1 - nOut);
    currentLayerNeurons[neuronIndex].Dedz = currentDeDz;

    //просчет изменений в весах нейронов текущего слоя по состоянию нейронов предыдущего
    for (int currentWeightIndex = 0; currentWeightIndex < previousLayerNeuronCount; ++currentWeightIndex)
    {
        float prevOut =
            previousLayerNeurons[currentWeightIndex].LastState;

        float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
        float coef = prevOut + regularizationCoef;
        float n = learningRate * currentDeDz * coef;

        nabla[currentNablaIndex + currentWeightIndex] += n;
    }
}

__kernel void OutputLayerTrain(
    __global Neuron * previousLayerNeurons,
    __global Neuron * currentLayerNeurons,

    __global float * desiredOutput,

    __global float * currentLayerWeights,
            
    __global float * nabla,

    int previousLayerNeuronCount,
    int currentLayerNeuronCount,

    float learningRate,
    float regularizationFactor,
    float dataCount)
{
    int neuronIndex = get_global_id(0);

    float nOut = currentLayerNeurons[neuronIndex].LastState;

    float n =
        //rOut * (1 - nOut)
        <firstDerivative_nOut>
        * (desiredOutput[neuronIndex] - nOut);

    currentLayerNeurons[neuronIndex].Dedz = n;

    int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

    for (int weightIndex = 0; weightIndex < previousLayerNeuronCount; ++weightIndex)
    {
        float deltaWeight =
            learningRate *
            n *
            (previousLayerNeurons[weightIndex].LastState
                + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

        nabla[nablaNeuronShift + weightIndex] += deltaWeight;
    }
}
";
        private string _additionalKernelsSource = @"
__kernel void ClearNabla(
    __global float * nabla,
    int count, //общее количество флоатов для обработки (для всех кернелов, длина currentLayerWeights, длина nabla)
    int kernelDataCount) //количество флоатов для обработки ОДНИМ кернелом (должно быть кратно 4м!!!)
{
    //__constant
        float4 zero = 0;

    int kernelIndex = get_global_id(0);
    
    int d1StartIndex = kernelIndex * kernelDataCount;
    int d1Count = min(kernelDataCount, count - d1StartIndex);

    int d4StartIndex = d1StartIndex / 4;
    int d4Count = d1Count / 4;
    
    int d1StartRemainder = d1StartIndex + d4Count * 4;

    for(int cc = d4StartIndex; cc < d4StartIndex + d4Count; cc++)
    {
        vstore4(
            zero,
            cc,
            nabla);
    }

    for(int cc = d1StartRemainder; cc < d1StartIndex + d1Count; cc++)
    {
        nabla[cc] = 0;
    }
}

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

        float4 result = currentLayerWeights4 + nabla4;

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

