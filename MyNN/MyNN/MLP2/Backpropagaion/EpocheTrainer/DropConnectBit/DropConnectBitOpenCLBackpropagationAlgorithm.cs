using System;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit.WeightMask;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit
{
    public class DropConnectBitOpenCLBackpropagationAlgorithm<T> : IEpocheTrainer
        where T : ILayerInference
    {
        private readonly IRandomizer _randomizer;
        private readonly MLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly float _p;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        private IOpenCLWeightBitMaskContainer _weightMask;

        private readonly DropConnectBitOpenCLForwardPropagation _dropConnectForwardPropagation;

        public IForwardPropagation ForwardPropagation
        {
            get;
            private set;
        }

        public DropConnectBitOpenCLBackpropagationAlgorithm(
            IRandomizer randomizer,
            VectorizationSizeEnum vse,
            MLP mlp,
            ILearningAlgorithmConfig config,
            CLProvider clProvider,
            int sampleCount = 5000,
            float p = 0.5f
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _randomizer = randomizer;
            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;
            _sampleCount = sampleCount;
            _p = p;

            this.PrepareInfrastructure();

            _dropConnectForwardPropagation = new DropConnectBitOpenCLForwardPropagation(
                vse,
                _mlp,
                _clProvider,
                this._weightMask);

            ForwardPropagation = new InferenceOpenCLForwardPropagation<T>(
                vse,
                _mlp,
                _clProvider,
                _randomizer,
                _sampleCount,
                _p);

        }

        #region prepare infrastructure

        private void PrepareInfrastructure()
        {
            GenerateMems();

            //загружаем программу и параметры
            LoadPrograms();

            CreateMasks();
        }

        private void CreateMasks()
        {
            _weightMask = new BigArrayOpenCLWeightBitMaskContainer(
                _clProvider,
                _mlp,
                _randomizer,
                _p);
        }

        private void GenerateMems()
        {
            _nablaWeights = new MemFloat[_mlp.Layers.Length];
            _deDz = new MemFloat[_mlp.Layers.Length];
        }

        private void LoadPrograms()
        {
            var kg = new DropConnectBitKernelConstructor(
                _mlp,
                _config);

            _hiddenKernelIncrement = new Kernel[_mlp.Layers.Length];
            _hiddenKernelOverwrite = new Kernel[_mlp.Layers.Length];
            _outputKernelIncrement = new Kernel[_mlp.Layers.Length];
            _outputKernelOverwrite = new Kernel[_mlp.Layers.Length];

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                _hiddenKernelIncrement[layerIndex] = _clProvider.CreateKernel(
                    kg.GetIncrementCalculationKernelsSource(layerIndex),
                    "HiddenLayerTrain");
                
                _hiddenKernelOverwrite[layerIndex] = _clProvider.CreateKernel(
                    kg.GetOverwriteCalculationKernelsSource(layerIndex),
                    "HiddenLayerTrain");

                _outputKernelIncrement[layerIndex] = _clProvider.CreateKernel(
                    kg.GetIncrementCalculationKernelsSource(layerIndex),
                    "OutputLayerTrain");
                
                _outputKernelOverwrite[layerIndex] = _clProvider.CreateKernel(
                    kg.GetOverwriteCalculationKernelsSource(layerIndex),
                    "OutputLayerTrain");
            }

            //определяем кернел обновления весов
            _updateWeightKernel = _clProvider.CreateKernel(
                DropConnectBitKernelConstructor.UpdateWeightKernelSource,
                "UpdateWeightKernel");
        }

        #endregion

        public void PreTrainInit(DataSet data)
        {
            //создаем массивы смещений по весам и dedz
            for (var i = 1; i < _mlp.Layers.Length; i++)
            {
                var lastLayer = i == (_mlp.Layers.Length - 1);
                var biasNeuronCount = lastLayer ? 0 : 1;

                _nablaWeights[i] = _clProvider.CreateFloatMem(
                    (_mlp.Layers[i].Neurons.Length - biasNeuronCount) * _mlp.Layers[i].Neurons[0].Weights.Length,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

                _deDz[i] = _clProvider.CreateFloatMem(
                    _mlp.Layers[i].NonBiasNeuronCount,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            }

            var outputLength = _mlp.Layers.Last().NonBiasNeuronCount;

            //создаем объекты желаемых выходных данных (т.е. правильных ответов сети)
            _desiredOutput = _clProvider.CreateFloatMem(
                outputLength,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);
        }

        public void TrainEpoche(
            DataSet data, 
            string epocheRoot, 
            float learningRate)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }

            #region one epoche

            //переносим веса сети в объекты OpenCL
            //_clProvider.Unpack();
            _dropConnectForwardPropagation.PushWeights();

            //гоним на устройство
            foreach (var nw in _nablaWeights)
            {
                if (nw != null)
                {
                    nw.Write(BlockModeEnum.NonBlocking);
                }
            }

            _dropConnectForwardPropagation.ClearAndPushHiddenLayers();

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //process data set
            var currentIndex = 0;
            do
            {
                //process one batch
                for (int batchIndex = currentIndex, inBatchIndex = 0; batchIndex < currentIndex + _config.BatchSize && batchIndex < data.Count; ++batchIndex, ++inBatchIndex)
                {
                    //set next weight mask index
                    this._weightMask.RegenerateMask();

                    // Make sure we're done with everything that's been requested before
                    _clProvider.QueueFinish();

                    //train data
                    var trainData = data[batchIndex];

                    //---------------------------- forward pass ----------------------------

                    _dropConnectForwardPropagation.Propagate(trainData);

                    //---------------------------- backward pass, error propagation ----------------------------

                    //отправляем на OpenCL желаемые выходы
                    trainData.Output.CopyTo(_desiredOutput.Array, 0);
                    _desiredOutput.Write(BlockModeEnum.Blocking);

                    //output layer
                    var outputLayerIndex = _mlp.Layers.Length - 1;

                    var outputLayer = _mlp.Layers[outputLayerIndex];
                    var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

                    if (inBatchIndex == 0)
                    {
                        _outputKernelOverwrite.Last()
                            .SetKernelArgMem(0, _dropConnectForwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _dropConnectForwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _dropConnectForwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _dropConnectForwardPropagation.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(6, outputNablaLayer)
                            .SetKernelArgMem(7, _weightMask.MaskMem[outputLayerIndex])
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(9, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(10, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(11, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(12, 4, learningRate)
                            .SetKernelArg(13, 4, _config.RegularizationFactor)
                            .SetKernelArg(14, 4, (float)(data.Count))
                            .SetKernelArg(15, 4, _weightMask.BitMask)
                            .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                    }
                    else
                    {
                        _outputKernelIncrement.Last()
                            .SetKernelArgMem(0, _dropConnectForwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _dropConnectForwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _dropConnectForwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _dropConnectForwardPropagation.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(6, outputNablaLayer)
                            .SetKernelArgMem(7, _weightMask.MaskMem[outputLayerIndex])
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(9, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(10, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(11, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(12, 4, learningRate)
                            .SetKernelArg(13, 4, _config.RegularizationFactor)
                            .SetKernelArg(14, 4, (float)(data.Count))
                            .SetKernelArg(15, 4, _weightMask.BitMask)
                            .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                    }

                    
                    //hidden layers
                    //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                    //тут паралеллизовать нечего
                    for (int hiddenLayerIndex = _mlp.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                    {
                        //определяем слои
                        var prevLayer = _mlp.Layers[hiddenLayerIndex - 1];
                        var currentLayer = _mlp.Layers[hiddenLayerIndex];
                        var nextLayer = _mlp.Layers[hiddenLayerIndex + 1];

                        if (inBatchIndex == 0)
                        {
                            _hiddenKernelOverwrite[hiddenLayerIndex]
                                .SetKernelArgMem(0, _dropConnectForwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _dropConnectForwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _dropConnectForwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _dropConnectForwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _dropConnectForwardPropagation.WeightMem[hiddenLayerIndex + 1])
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArgMem(8, _weightMask.MaskMem[hiddenLayerIndex])
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(11, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(12, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(14, 4, learningRate)
                                .SetKernelArg(15, 4, _config.RegularizationFactor)
                                .SetKernelArg(16, 4, (float)(data.Count))
                                .SetKernelArg(17, 4, _weightMask.BitMask)
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                        }
                        else
                        {
                            _hiddenKernelIncrement[hiddenLayerIndex]
                                .SetKernelArgMem(0, _dropConnectForwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _dropConnectForwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _dropConnectForwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _dropConnectForwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _dropConnectForwardPropagation.WeightMem[hiddenLayerIndex + 1])
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArgMem(8, _weightMask.MaskMem[hiddenLayerIndex])
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(11, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(12, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(14, 4, learningRate)
                                .SetKernelArg(15, 4, _config.RegularizationFactor)
                                .SetKernelArg(16, 4, (float)(data.Count))
                                .SetKernelArg(17, 4, _weightMask.BitMask)
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                        }
                    }
                    //*/

                    //// Make sure we're done with everything that's been requested before
                    //_clProvider.QueueFinish();

                    int logStep = data.Count / 100;
                    if (logStep > 0 && currentIndex % logStep == 0)
                    {
                        ConsoleAmbientContext.Console.Write(
                            "Epoche progress: {0}%, {1}      ",
                            (currentIndex * 100 / data.Count),
                            DateTime.Now.ToString());

                        ConsoleAmbientContext.Console.ReturnCarriage();
                    }
                }

                //update weights and bias into opencl memory wrappers

                for (int layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                {
                    var weightMem = _dropConnectForwardPropagation.WeightMem[layerIndex];
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

            ConsoleAmbientContext.Console.Write(new string(' ', 60));
            ConsoleAmbientContext.Console.ReturnCarriage();

            //конец эпохи обучения

            //считываем веса с устройства
            foreach (var wm in _dropConnectForwardPropagation.WeightMem)
            {
                if (wm != null)
                {
                    wm.Read(BlockModeEnum.Blocking);
                }
            }

            //write new weights and biases into network
            for (int layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var weightLayer = _dropConnectForwardPropagation.WeightMem[layerIndex];

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

    }
}
