using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic2.OpenCL.CPU
{
    /// <summary>
    /// Inefficient implementation of classic backpropagation with transposed weights.
    /// It seems to be OBSOLETE today.
    /// </summary>
    public class CPUTranspose2EpocheTrainer : IEpocheTrainer
    {
        private readonly IMemLayerContainer[] _containers;
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private MemFloat[] _transposers;

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
        /// <param name="vse">Vectorization mode</param>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="config">Learning config</param>
        /// <param name="clProvider">OpenCL provider</param>
        public CPUTranspose2EpocheTrainer(
            VectorizationSizeEnum vse,
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

            if (clProvider.ChoosedDeviceType == DeviceType.Cpu)
            {
                ConsoleAmbientContext.Console.WriteLine("========================================= WARNING =========================================");
                ConsoleAmbientContext.Console.WriteLine("This algorithm are so slow on CPU; it should test this algorithm on GPU hardware and delete it if it will not be inferior than the default.");
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;

            var cc = new CPUPropagatorComponentConstructor(
                _clProvider,
                vse
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

            _transposers = new MemFloat[_mlp.Layers.Length];
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
                "UpdateWeightAndTransposedWeightsKernel");
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

                _transposers[i] = _clProvider.CreateFloatMem(
                    _mlp.Layers[i - 1].Neurons.Length * _mlp.Layers[i].NonBiasNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

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

            //очищаем и гоним на устройство
            foreach (var nw in _nablaWeights)
            {
                if (nw != null)
                {
                    Array.Clear(nw.Array, 0, nw.Array.Length);
                    nw.Write(BlockModeEnum.NonBlocking);
                }
            }

            _forwardPropagation.ClearAndPushHiddenLayers();

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //транспонируем все веса вначале (так как повторно они транспонируются после батча)
            //обычным методом обновления (при nabla равной нулю везде, таким образом
            //веса не меняются)
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var weightMem = _containers[layerIndex].WeightMem;
                var nablaMem = _nablaWeights[layerIndex];

                //обновляем веса и транспонированные веса
                _updateWeightKernel
                    .SetKernelArgMem(0, weightMem)
                    .SetKernelArgMem(1, nablaMem)
                    .SetKernelArgMem(2, _transposers[layerIndex])
                    .SetKernelArg(3, 4, _mlp.Layers[layerIndex].NonBiasNeuronCount)
                    .SetKernelArg(4, 4, _mlp.Layers[layerIndex - 1].Neurons.Length)
                    .SetKernelArg(5, 4, (float)(_config.BatchSize))
                    .EnqueueNDRangeKernel(
                        _mlp.Layers[layerIndex].NonBiasNeuronCount,
                        _mlp.Layers[layerIndex - 1].Neurons.Length);
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
                    var trainDataItem = data[batchIndex];

                    //---------------------------- forward pass ----------------------------

                    _forwardPropagation.Propagate(trainDataItem);

                    //---------------------------- backward pass, error propagation ----------------------------

                    //отправляем на OpenCL желаемые выходы
                    trainDataItem.Output.CopyTo(_desiredOutput.Array, 0);
                    _desiredOutput.Write(BlockModeEnum.Blocking);

                    //output layer
                    var outputLayerIndex = _mlp.Layers.Length - 1;

                    var outputLayer = _mlp.Layers[outputLayerIndex];
                    var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

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
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(9, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(10, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(11, 4, learningRate)
                            .SetKernelArg(12, 4, _config.RegularizationFactor)
                            .SetKernelArg(13, 4, (float)(data.Count))
                            .EnqueueNDRangeKernel(outputLayer.NonBiasNeuronCount);
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
                            .SetKernelArg(7, 4, preOutputLayer.Neurons.Length / 4)
                            .SetKernelArg(8, 4, preOutputLayer.Neurons.Length - (preOutputLayer.Neurons.Length % 4))
                            .SetKernelArg(9, 4, preOutputLayer.Neurons.Length)
                            .SetKernelArg(10, 4, outputLayer.NonBiasNeuronCount)
                            .SetKernelArg(11, 4, learningRate)
                            .SetKernelArg(12, 4, _config.RegularizationFactor)
                            .SetKernelArg(13, 4, (float)(data.Count))
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
                                .SetKernelArgMem(0, _containers[hiddenLayerIndex].NetMem)
                                .SetKernelArgMem(1, _containers[hiddenLayerIndex - 1].StateMem)
                                .SetKernelArgMem(2, _containers[hiddenLayerIndex].StateMem)
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _containers[hiddenLayerIndex].WeightMem)
                                .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1])
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(11, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(12, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, learningRate)
                                .SetKernelArg(14, 4, _config.RegularizationFactor)
                                .SetKernelArg(15, 4, (float)(data.Count))
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
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
                                .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1])
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArg(8, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(11, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(12, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, learningRate)
                                .SetKernelArg(14, 4, _config.RegularizationFactor)
                                .SetKernelArg(15, 4, (float)(data.Count))
                                .EnqueueNDRangeKernel(currentLayer.NonBiasNeuronCount);
                        }
                    }

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
                    var weightMem = _containers[layerIndex].WeightMem;
                    var nablaMem = _nablaWeights[layerIndex];

                    //обновляем веса и транспонированные веса
                    _updateWeightKernel
                        .SetKernelArgMem(0, weightMem)
                        .SetKernelArgMem(1, nablaMem)
                        .SetKernelArgMem(2, _transposers[layerIndex])
                        .SetKernelArg(3, 4, _mlp.Layers[layerIndex].NonBiasNeuronCount)
                        .SetKernelArg(4, 4, _mlp.Layers[layerIndex - 1].Neurons.Length)
                        .SetKernelArg(5, 4, (float)(_config.BatchSize))
                        .EnqueueNDRangeKernel(
                            new int[]
                            {
                                _mlp.Layers[layerIndex].NonBiasNeuronCount,
                                _mlp.Layers[layerIndex - 1].Neurons.Length
                            });
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

    }
}
