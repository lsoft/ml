using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
using MyNN.MLP.Classic.Transposer;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU
{
    /// <summary>
    /// Classic backpropagation epoche trainer that enables GPU-OpenCL with transposed weights.
    /// This implementation of classic backpropagation is optimized for training in batch-mode.
    /// </summary>
    public class GPUTransposeEpocheTrainer : IEpocheTrainer
    {
        private readonly IMemLayerContainer[] _containers;
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private IOpenCLTransposer[] _transposers;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        private readonly MLP.ForwardPropagation.ForwardPropagation _forwardPropagation;
        public IForwardPropagation ForwardPropagation
        {
            get
            {
                return
                    _forwardPropagation;
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="config">Learning config</param>
        /// <param name="clProvider">OpenCL provider</param>
        public GPUTransposeEpocheTrainer(
            IMLP mlp,
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

            if (config.BatchSize == 1)
            {
                ConsoleAmbientContext.Console.WriteWarning("This backpropagation algorithm optimized to work in batch mode (typical with batch size = [25;100]). Online backpropagation is not an optimal choise. Try to use CPUBackpropagationEpocheTrainer.");
            }

            if (config.BatchSize > 1 && config.BatchSize < 25)
            {
                ConsoleAmbientContext.Console.WriteWarning("Too low minibatch size ({0}) to achieve optimal performance. Try to increase batch size to 25 minimum.", config.BatchSize);
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;

            var cc = new GPUPropagatorComponentConstructor(
                _clProvider
                );

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            cc.CreateComponents(
                mlp,
                out containers,
                out propagators);

            _containers = containers.ToList().ConvertAll(j => j as IMemLayerContainer).ToArray();

            _forwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                containers,
                propagators,
                _mlp
                );

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
            var kg = new KernelConstructor(
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
                KernelConstructor.UpdateWeightKernelSource,
                "UpdateWeightKernel");
        }

        #endregion

        public void PreTrainInit(IDataSet data)
        {
            //создаем инфраструктуру
            for (var i = 1; i < _mlp.Layers.Length; i++)
            {
                var lastLayer = i == (_mlp.Layers.Length - 1);
                var biasNeuronCount = lastLayer ? 0 : 1;

                _nablaWeights[i] = _clProvider.CreateFloatMem(
                    (_mlp.Layers[i].Neurons.Length - biasNeuronCount) * _mlp.Layers[i].Neurons[0].Weights.Length,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                _deDz[i] = _clProvider.CreateFloatMem(
                    _mlp.Layers[i].NonBiasNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                _transposers[i] = new TransposerNvidia(
                    _clProvider,
                    _containers[i].WeightMem,
                    _mlp.Layers[i - 1].Neurons.Length,
                    _mlp.Layers[i].NonBiasNeuronCount);
            }

            var outputLength = _mlp.Layers.Last().NonBiasNeuronCount;

            //создаем объекты желаемых выходных данных (т.е. правильных ответов сети)
            _desiredOutput = _clProvider.CreateFloatMem(
                outputLength,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);
        }

        public void TrainEpoche(
            IDataSet data,
            IArtifactContainer artifactContainer, 
            float learningRate)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }

            #region one epoche

            //переносим веса сети в объекты OpenCL
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
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                _transposers[layerIndex].Transpose();
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //process data set
            var enumerator = data.StartIterate();
            try
            {
                var allowedToContinue = true;
                for (
                    var currentIndex = 0;
                    allowedToContinue;
                    currentIndex += _config.BatchSize
                    )
                {
                    //process one batch
                    for (
                        var inBatchIndex = 0;
                        inBatchIndex < _config.BatchSize && allowedToContinue;
                        ++inBatchIndex
                        )
                    {
                        allowedToContinue = enumerator.MoveNext();
                        if (allowedToContinue)
                        {
                            //train data
                            var trainDataItem = enumerator.Current;

                            #region forward pass

                            _forwardPropagation.Propagate(trainDataItem);

                            #endregion

                            #region backward pass, error propagation

                            //отправляем на OpenCL желаемые выходы
                            trainDataItem.Output.CopyTo(_desiredOutput.Array, 0);
                            _desiredOutput.Write(BlockModeEnum.Blocking);

                            #region output layer

                            var outputLayerIndex = _mlp.Layers.Length - 1;

                            var outputLayer = _mlp.Layers[outputLayerIndex];
                            var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                            var outputNablaLayer = _nablaWeights[outputLayerIndex];

                            const uint OutputLocalGroupSize = 128;
                            uint OutputGlobalGroupSize =
                                (uint) outputLayer.NonBiasNeuronCount*OutputLocalGroupSize;

                            if (inBatchIndex == 0)
                            {
                                _outputKernelOverwrite.Last()
                                    .SetKernelArgMem(0, _containers[outputLayerIndex].NetMem)
                                    .SetKernelArgMem(1, _containers[outputLayerIndex - 1].StateMem)
                                    .SetKernelArgMem(2, _containers[outputLayerIndex].StateMem)
                                    .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                                    .SetKernelArgMem(4, _desiredOutput)
                                    .SetKernelArgMem(5, _containers[outputLayerIndex].WeightMem)
                                    .SetKernelArgMem(6, outputNablaLayer)
                                    .SetKernelArg(7, 4, preOutputLayer.Neurons.Length)
                                    .SetKernelArg(8, 4, outputLayer.NonBiasNeuronCount)
                                    .SetKernelArg(9, 4, learningRate)
                                    .SetKernelArg(10, 4, _config.RegularizationFactor)
                                    .SetKernelArg(11, 4, (float) (data.Count))
                                    //.EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount)
                                    .EnqueueNDRangeKernel(
                                        new[]
                                        {
                                            OutputGlobalGroupSize
                                        },
                                        new[]
                                        {
                                            OutputLocalGroupSize
                                        })
                                    ;
                            }
                            else
                            {
                                _outputKernelIncrement.Last()
                                    .SetKernelArgMem(0, _containers[outputLayerIndex].NetMem)
                                    .SetKernelArgMem(1, _containers[outputLayerIndex - 1].StateMem)
                                    .SetKernelArgMem(2, _containers[outputLayerIndex].StateMem)
                                    .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                                    .SetKernelArgMem(4, _desiredOutput)
                                    .SetKernelArgMem(5, _containers[outputLayerIndex].WeightMem)
                                    .SetKernelArgMem(6, outputNablaLayer)
                                    .SetKernelArg(7, 4, preOutputLayer.Neurons.Length)
                                    .SetKernelArg(8, 4, outputLayer.NonBiasNeuronCount)
                                    .SetKernelArg(9, 4, learningRate)
                                    .SetKernelArg(10, 4, _config.RegularizationFactor)
                                    .SetKernelArg(11, 4, (float) (data.Count))
                                    //.EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount)
                                    .EnqueueNDRangeKernel(
                                        new[]
                                        {
                                            OutputGlobalGroupSize
                                        },
                                        new[]
                                        {
                                            OutputLocalGroupSize
                                        })
                                    ;
                            }

                            #endregion

                            #region hidden layers

                            //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                            //тут паралеллизовать нечего
                            for (int hiddenLayerIndex = _mlp.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                            {
                                //определяем слои
                                var prevLayer = _mlp.Layers[hiddenLayerIndex - 1];
                                var currentLayer = _mlp.Layers[hiddenLayerIndex];
                                var nextLayer = _mlp.Layers[hiddenLayerIndex + 1];

                                const uint HiddenLocalGroupSize = 32;
                                uint HiddenGlobalGroupSize =
                                    (uint) currentLayer.NonBiasNeuronCount*HiddenLocalGroupSize
                                    ;

                                if (inBatchIndex == 0)
                                {
                                    _hiddenKernelOverwrite[hiddenLayerIndex]
                                        .SetKernelArgMem(0, _containers[hiddenLayerIndex].NetMem)
                                        .SetKernelArgMem(1, _containers[hiddenLayerIndex - 1].StateMem)
                                        .SetKernelArgMem(2, _containers[hiddenLayerIndex].StateMem)
                                        .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                        .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                        .SetKernelArgMem(5, _containers[hiddenLayerIndex].WeightMem)
                                        .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1].Destination)
                                        .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                        .SetKernelArg(8, 4, prevLayer.Neurons.Length)
                                        .SetKernelArg(9, 4, currentLayer.NonBiasNeuronCount)
                                        .SetKernelArg(10, 4, nextLayer.NonBiasNeuronCount)
                                        .SetKernelArg(11, 4, learningRate)
                                        .SetKernelArg(12, 4, _config.RegularizationFactor)
                                        .SetKernelArg(13, 4, (float) (data.Count))
                                        .SetKernelArgLocalMem(14, 4*HiddenLocalGroupSize)
                                        //.EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount)
                                        .EnqueueNDRangeKernel(
                                            new[]
                                            {
                                                HiddenGlobalGroupSize
                                            },
                                            new[]
                                            {
                                                HiddenLocalGroupSize
                                            })
                                        ;
                                }
                                else
                                {
                                    _hiddenKernelIncrement[hiddenLayerIndex]
                                        .SetKernelArgMem(0, _containers[hiddenLayerIndex].NetMem)
                                        .SetKernelArgMem(1, _containers[hiddenLayerIndex - 1].StateMem)
                                        .SetKernelArgMem(2, _containers[hiddenLayerIndex].StateMem)
                                        .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                        .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                        .SetKernelArgMem(5, _containers[hiddenLayerIndex].WeightMem)
                                        .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1].Destination)
                                        .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                        .SetKernelArg(8, 4, prevLayer.Neurons.Length)
                                        .SetKernelArg(9, 4, currentLayer.NonBiasNeuronCount)
                                        .SetKernelArg(10, 4, nextLayer.NonBiasNeuronCount)
                                        .SetKernelArg(11, 4, learningRate)
                                        .SetKernelArg(12, 4, _config.RegularizationFactor)
                                        .SetKernelArg(13, 4, (float) (data.Count))
                                        .SetKernelArgLocalMem(14, 4*HiddenLocalGroupSize)
                                        //.EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount)
                                        .EnqueueNDRangeKernel(
                                            new[]
                                            {
                                                HiddenGlobalGroupSize
                                            },
                                            new[]
                                            {
                                                HiddenLocalGroupSize
                                            })
                                        ;
                                }
                            }

                            #endregion

                            #region logging

                            var logStep = data.Count/100;
                            if (logStep > 0 && currentIndex%logStep == 0)
                            {
                                ConsoleAmbientContext.Console.Write(
                                    "Epoche progress: {0}%, {1}      ",
                                    ((long)currentIndex * 100 / data.Count),
                                    DateTime.Now.ToString());

                                ConsoleAmbientContext.Console.ReturnCarriage();
                            }

                            #endregion

                            #endregion
                        }
                    }

                    #region update weights and bias into opencl memory wrappers

                    for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                    {
                        var weightMem = _containers[layerIndex].WeightMem;
                        var nablaMem = _nablaWeights[layerIndex];

                        //обновляем веса
                        _updateWeightKernel
                            .SetKernelArgMem(0, weightMem)
                            .SetKernelArgMem(1, nablaMem)
                            .SetKernelArg(2, 4, (float)(_config.BatchSize))
                            .SetKernelArg(3, 4, weightMem.Array.Length)
                            .EnqueueNDRangeKernel(weightMem.Array.Length);

                        //транспонируем
                        _transposers[layerIndex].Transpose();
                    }

                    #endregion

                    // Make sure we're done with everything that's been requested before
                    _clProvider.QueueFinish();
                }
            }
            finally
            {
                enumerator.Dispose();
            }

            #endregion

            ConsoleAmbientContext.Console.Write(new string(' ', 60));
            ConsoleAmbientContext.Console.ReturnCarriage();

            //конец эпохи обучения

            //считываем веса с устройства
            foreach (var container in _containers)
            {
                if (container.WeightMem != null)
                {
                    container.WeightMem.Read(BlockModeEnum.Blocking);
                }
            }

            //write new weights and biases into network
            for (int layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var weightLayer = _containers[layerIndex].WeightMem;

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

        #region private code

        private void CheckEquals(
            float[] source,
            float[] transposed,
            int width,
            int height)
        {
            if (source == null)
            {
                throw new ArgumentNullException("source");
            }
            if (transposed == null)
            {
                throw new ArgumentNullException("transposed");
            }

            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    var s = source[h * width + w];
                    var d = transposed[w * height + h];

                    var diff = s - d;

                    if (Math.Abs(diff) >= float.Epsilon)
                    {
                        throw
                            new Exception(diff.ToString());
                    }
                }
            }
        }

        private void ConsoleDump(string name, float[] body, int width, int height)
        {
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (body == null)
            {
                throw new ArgumentNullException("body");
            }

            ConsoleAmbientContext.Console.WriteLine(name);

            for (var h = 0; h < height; h++)
            {
                var listw = new List<float>();
                for (var w = 0; w < width; w++)
                {
                    listw.Add(body[h * width + w]);
                }

                var s = string.Join(
                    " ",
                    listw.ConvertAll(j => DoubleConverter.ToExactString(j)).ToArray());

                ConsoleAmbientContext.Console.WriteLine(s);
            }
        }

        #endregion
    }
}
