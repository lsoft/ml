﻿using System;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Transposer;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL
{
    public class GPUTransposeBackpropagationAlgorithm : IEpocheTrainer
    {
        private readonly MLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private IOpenCLTransposer[] _transposers;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        private readonly GPUForwardPropagation _forwardPropagation;
        public IForwardPropagation ForwardPropagation
        {
            get
            {
                return
                    _forwardPropagation;
            }
        }

        public GPUTransposeBackpropagationAlgorithm(
            MLP mlp,
            ILearningAlgorithmConfig config,
            CLProvider clProvider
            )
        {
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

            if (config.RegularizationFactor > float.Epsilon)
            {
                throw new NotSupportedException("config.RegularizationFactor > float.Epsilon");
            }

            if (config.BatchSize == 1)
            {
                ConsoleAmbientContext.Console.WriteWarning("This backpropagation algorithm optimized to work in batch mode (typical with batch size = [25;100]). Online backpropagation is not an optimal choise. Try to use OpenCLTranspose2BackpropagationAlgorithm.");
            }

            if (config.BatchSize > 1 && config.BatchSize < 25)
            {
                ConsoleAmbientContext.Console.WriteWarning("Too low minibatch size ({0}) to achieve optimal performance. Try to increase batch size to 25 minimum.", config.BatchSize);
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;

            _forwardPropagation = new GPUForwardPropagation(
                _mlp,
                _clProvider);

            this.PrepareInfrastructure();
        }

        #region prepare infrastructure

        private void PrepareInfrastructure()
        {

            GenerateMems();

            //загружаем программу и параметры
            LoadPrograms();
            
        }

        private void GenerateMems()
        {
            _nablaWeights = new MemFloat[_mlp.Layers.Length];
            _deDz = new MemFloat[_mlp.Layers.Length];
            _transposers = new IOpenCLTransposer[_mlp.Layers.Length];
        }

        private void LoadPrograms()
        {
            var kg = new GPUTransposeKernelConstructor(
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
                GPUKernelConstructor.UpdateWeightKernelSource,
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

                if (i > 1)
                {
                    //для ПЕРВОГО СКРЫТОГО слоя не надо траспонера

                    _transposers[i] = new TransposerNvidia(
                        _clProvider,
                        _forwardPropagation.WeightMem[i],
                        _mlp.Layers[i - 1].Neurons.Length,
                        _mlp.Layers[i].NonBiasNeuronCount);
                }
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
            _forwardPropagation.PushWeights();

            //гоним на устройство
            foreach (var nw in _nablaWeights)
            {
                if (nw != null)
                {
                    nw.Write(BlockModeEnum.NonBlocking);
                }
            }

            _forwardPropagation.ClearAndPushHiddenLayers();

            //транспонируем все веса вначале (так как повторно они транспонируются после батча)
            //ПЕРВЫЙ СКРЫТЫЙ слой транспонировать не надо
            for (var layerIndex = 2; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                _transposers[layerIndex].Transpose();
            }

            //process data set
            var currentIndex = 0;
            do
            {
                //process one batch

                for (int batchIndex = currentIndex, inBatchIndex = 0; batchIndex < currentIndex + _config.BatchSize && batchIndex < data.Count; ++batchIndex, ++inBatchIndex)
                {
                    //train data
                    var trainDataItem = data[batchIndex];

                    //---------------------------- forward pass ----------------------------

                    _forwardPropagation.Propagate(trainDataItem);

                    //---------------------------- backward pass, error propagation ----------------------------

                    //отправляем на OpenCL желаемые выходы
                    trainDataItem.Output.CopyTo(_desiredOutput.Array, 0);
                    _desiredOutput.Write(BlockModeEnum.NonBlocking);

                    //output layer
                    var outputLayerIndex = _mlp.Layers.Length - 1;

                    var outputLayer = _mlp.Layers[outputLayerIndex];
                    var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

                    const int OutputLocalGroupSize = 256;
                    int OutputGlobalGroupSize = 128 * _clProvider.Parameters.NumComputeUnits * OutputLocalGroupSize;

                    if (inBatchIndex == 0)
                    {
                        //_forwardPropagation.NetMem[outputLayerIndex].Read(BlockModeEnum.Blocking);
                        //_clProvider.QueueFinish();

                        _outputKernelOverwrite.Last()
                            .SetKernelArgMem(0, _forwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _forwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _forwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _forwardPropagation.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(6, outputNablaLayer)
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(8, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(9, 4, learningRate)
                            .SetKernelArg(10, 4, _config.RegularizationFactor)
                            .SetKernelArg(11, 4, (float)(data.Count))
                            //.EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                            .EnqueueNDRangeKernel(
                                new int[]
                                {
                                    OutputGlobalGroupSize
                                },
                                new int[]
                                {
                                    OutputLocalGroupSize
                                });
                    }
                    else
                    {
                        _outputKernelIncrement.Last()
                            .SetKernelArgMem(0, _forwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _forwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _forwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _forwardPropagation.WeightMem[outputLayerIndex])
                            .SetKernelArgMem(6, outputNablaLayer)
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(8, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(9, 4, learningRate)
                            .SetKernelArg(10, 4, _config.RegularizationFactor)
                            .SetKernelArg(11, 4, (float)(data.Count))
                            //.EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
                            .EnqueueNDRangeKernel(
                                new int[]
                                {
                                    OutputGlobalGroupSize
                                },
                                new int[]
                                {
                                    OutputLocalGroupSize
                                });

                    }

                    
                    //hidden layers
                    //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                    //тут паралеллизовать нечего
                    for (var hiddenLayerIndex = _mlp.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                    {
                        //определяем слои
                        var prevLayer = _mlp.Layers[hiddenLayerIndex - 1];
                        var currentLayer = _mlp.Layers[hiddenLayerIndex];
                        var nextLayer = _mlp.Layers[hiddenLayerIndex + 1];

                        const int HiddenLocalGroupSize = 128;
                        int HiddenGlobalGroupSize = 8 * _clProvider.Parameters.NumComputeUnits * HiddenLocalGroupSize;

                        if (inBatchIndex == 0)
                        {
                            _hiddenKernelOverwrite[hiddenLayerIndex]
                                .SetKernelArgMem(0, _forwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _forwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _forwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _forwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1].Destination)
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(9, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(10, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(11, 4, learningRate)
                                .SetKernelArg(12, 4, _config.RegularizationFactor)
                                .SetKernelArg(13, 4, (float)(data.Count))
                                .SetKernelArgLocalMem(14, 4 * HiddenLocalGroupSize)
                                //.EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                                .EnqueueNDRangeKernel(
                                    new int[]
                                    {
                                        HiddenGlobalGroupSize
                                        //currentLayer.NonBiasNeuronCount + (LocalGroupSize -  currentLayer.NonBiasNeuronCount % LocalGroupSize)
                                    },
                                    new int[]
                                    {
                                        HiddenLocalGroupSize
                                        //LocalGroupSize
                                    });
                        }
                        else
                        {
                            _hiddenKernelIncrement[hiddenLayerIndex]
                                .SetKernelArgMem(0, _forwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _forwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _forwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _forwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1].Destination)
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(9, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(10, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(11, 4, learningRate)
                                .SetKernelArg(12, 4, _config.RegularizationFactor)
                                .SetKernelArg(13, 4, (float)(data.Count))
                                .SetKernelArgLocalMem(14, 4 * HiddenLocalGroupSize)
                                //.EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                                .EnqueueNDRangeKernel(
                                    new int[]
                                    {
                                        HiddenGlobalGroupSize
                                        //currentLayer.NonBiasNeuronCount + (LocalGroupSize -  currentLayer.NonBiasNeuronCount % LocalGroupSize)
                                    },
                                    new int[]
                                    {
                                        HiddenLocalGroupSize
                                        //LocalGroupSize
                                    });
                        }
                    }

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

                for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                {
                    var weightMem = _forwardPropagation.WeightMem[layerIndex];
                    var nablaMem = _nablaWeights[layerIndex];

                    var neuronCount = _mlp.Layers[layerIndex].NonBiasNeuronCount;
                    var weightCount = _mlp.Layers[layerIndex - 1].Neurons.Length;

                    //var localSize = 512;
                    //var globalSize = localSize*256;

                    _updateWeightKernel
                        .SetKernelArgMem(0, weightMem)
                        .SetKernelArgMem(1, nablaMem)
                        //.SetKernelArgLocalMem(2, 4 * weightCount)
                        .SetKernelArg(2, 4, (float)(_config.BatchSize))
                        .SetKernelArg(3, 4, neuronCount)
                        .SetKernelArg(4, 4, weightCount)
                        .SetKernelArg(5, 4, weightMem.Array.Length)
                        .EnqueueNDRangeKernel(weightMem.Array.Length);
                        //.EnqueueNDRangeKernel(
                        //    new int[]
                        //    {
                        //        globalSize  
                        //    }
                        //    , new int[]
                        //    {
                        //        localSize
                        //    }
                        //    );

                    //транспонируем
                    if (layerIndex > 1)
                    {
                        //ПЕРВЫЙ СКРЫТЫЙ слой не надо транспонировать
                        _transposers[layerIndex].Transpose();
                    }
                }

                //// Make sure we're done with everything that's been requested before
                //_clProvider.QueueFinish();

                currentIndex += _config.BatchSize;
            } while (currentIndex < data.Count);

            #endregion

            ConsoleAmbientContext.Console.Write(new string(' ', 60));
            ConsoleAmbientContext.Console.ReturnCarriage();

            //конец эпохи обучения

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //считываем веса с устройства
            foreach (var wm in _forwardPropagation.WeightMem)
            {
                if (wm != null)
                {
                    wm.Read(BlockModeEnum.Blocking);
                }
            }

            //_forwardPropagation.PopState();

            //write new weights and biases into network
            for (int layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var weightLayer = _forwardPropagation.WeightMem[layerIndex];

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