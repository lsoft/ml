using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.KernelText;
using MyNN.MLP.DropConnect.ForwardPropagation.Inference;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU;
using MyNN.MLP.DropConnect.Inferencer;
using MyNN.MLP.DropConnect.Inferencer.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU
{    
    
    /// <summary>
    /// Dropconnect backpropagation epoche trainer with bit mask that enables GPU-OpenCL
    /// </summary>
    public class DropConnectEpocheTrainer : IEpocheTrainer
    {
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        //форвардер с маской (для обучения сети)
        private readonly IMemLayerContainer[] _dropConnectContainers;
        private readonly IDropConnectLayerPropagator[] _dropConnectPropagators;
        private readonly MyNN.MLP.ForwardPropagation.ForwardPropagation _dropConnectForwardPropagation;

        //форвардер с "выводителем" (для итоговой аттестации сети)
        private IMemLayerContainer[] _inferenceContainers;
        private readonly MyNN.MLP.ForwardPropagation.ForwardPropagation _inferenceForwardPropagation;

        public IForwardPropagation ForwardPropagation
        {
            get
            {
                return
                    _inferenceForwardPropagation;
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="config">Learning config</param>
        /// <param name="maskContainerFactory">Фабрика битовых масок</param>
        /// <param name="inferencerFactory">Фабрика стохастического выводителя для слоя</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="p">Probability for each bit to be ONE (TRUE) (with p = 1 it completely disables mask and convert the model to classic backprop)</param>
        public DropConnectEpocheTrainer(
            IMLP mlp,
            ILearningAlgorithmConfig config,
            IOpenCLMaskContainerFactory maskContainerFactory,
            ILayerInferencerFactory inferencerFactory,
            CLProvider clProvider,
            float p = 0.5f
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
            if (maskContainerFactory == null)
            {
                throw new ArgumentNullException("maskContainerFactory");
            }
            if (inferencerFactory == null)
            {
                throw new ArgumentNullException("inferencerFactory");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;

            this.PrepareInfrastructure();

            #region создаем форвардер с маской (для обучения)
            
            {
                var cc = new PropagatorComponentConstructor(
                    _clProvider,
                    maskContainerFactory,
                    p
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                cc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators);

                _dropConnectContainers = containers.ToList().ConvertAll(j => j as IMemLayerContainer).ToArray();
                _dropConnectPropagators = propagators.ToList().ConvertAll(j => j as IDropConnectLayerPropagator).ToArray();

                _dropConnectForwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                    containers,
                    propagators,
                    _mlp
                    );
            }

            #endregion

            #region создаем форвардер с "выводителем" (для аттестации)

            {
                var cc = new ForwardPropagation.Inference.OpenCL.GPU.PropagatorComponentConstructor(
                    _clProvider,
                    inferencerFactory
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                cc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators);

                _inferenceContainers = containers.ToList().ConvertAll(j => j as IMemLayerContainer).ToArray();

                _inferenceForwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                    containers,
                    propagators,
                    _mlp
                    );
            }

            #endregion
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
        }

        private void LoadPrograms()
        {
            var kg = new KernelTextProvider(
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
                kg.UpdateWeightKernelSource,
                "UpdateWeightKernel");
        }

        #endregion

        public void PreTrainInit(IDataSet data)
        {
            //создаем массивы смещений по весам и dedz
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

                            _dropConnectForwardPropagation.Propagate(trainDataItem);

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

                            const int outputLocalSize = 128;
                            uint outputGlobalSize = outputLocalSize*(uint) outputLayer.NonBiasNeuronCount;

                            if (inBatchIndex == 0)
                            {
                                _outputKernelOverwrite.Last()
                                    .SetKernelArgMem(0, _dropConnectContainers[outputLayerIndex].NetMem)

                                    .SetKernelArgMem(1, _dropConnectContainers[outputLayerIndex - 1].StateMem)
                                    .SetKernelArgMem(2, _dropConnectContainers[outputLayerIndex].StateMem)
                                    .SetKernelArgMem(3, this._deDz[outputLayerIndex])

                                    .SetKernelArgMem(4, _desiredOutput)

                                    .SetKernelArgMem(5, _dropConnectContainers[outputLayerIndex].WeightMem)

                                    .SetKernelArgMem(6, outputNablaLayer)

                                    .SetKernelArgMem(7, _dropConnectPropagators[outputLayerIndex].MaskContainer.MaskMem)

                                    .SetKernelArg(8, 4, preOutputLayer.Neurons.Length)
                                    .SetKernelArg(9, 4, outputLayer.NonBiasNeuronCount)

                                    .SetKernelArg(10, 4, learningRate)
                                    .SetKernelArg(11, 4, _config.RegularizationFactor)
                                    .SetKernelArg(12, 4, (float) (data.Count))

                                    .SetKernelArg(13, 4, _dropConnectPropagators[outputLayerIndex].MaskContainer.BitMask)

                                    .EnqueueNDRangeKernel(
                                        new uint[]
                                        {
                                            outputGlobalSize
                                        }
                                        , new uint[]
                                        {
                                            outputLocalSize
                                        }
                                    );
                            }
                            else
                            {
                                _outputKernelIncrement.Last()
                                    .SetKernelArgMem(0, _dropConnectContainers[outputLayerIndex].NetMem)

                                    .SetKernelArgMem(1, _dropConnectContainers[outputLayerIndex - 1].StateMem)
                                    .SetKernelArgMem(2, _dropConnectContainers[outputLayerIndex].StateMem)
                                    .SetKernelArgMem(3, this._deDz[outputLayerIndex])

                                    .SetKernelArgMem(4, _desiredOutput)

                                    .SetKernelArgMem(5, _dropConnectContainers[outputLayerIndex].WeightMem)

                                    .SetKernelArgMem(6, outputNablaLayer)

                                    .SetKernelArgMem(7, _dropConnectPropagators[outputLayerIndex].MaskContainer.MaskMem)

                                    .SetKernelArg(8, 4, preOutputLayer.Neurons.Length)
                                    .SetKernelArg(9, 4, outputLayer.NonBiasNeuronCount)

                                    .SetKernelArg(10, 4, learningRate)
                                    .SetKernelArg(11, 4, _config.RegularizationFactor)
                                    .SetKernelArg(12, 4, (float) (data.Count))

                                    .SetKernelArg(13, 4, _dropConnectPropagators[outputLayerIndex].MaskContainer.BitMask)

                                    .EnqueueNDRangeKernel(
                                        new uint[]
                                        {
                                            outputGlobalSize
                                        }
                                        , new uint[]
                                        {
                                            outputLocalSize
                                        }
                                    );
                            }

                            #endregion

                            #region hidden layers

                            //цикл по скрытым слоям, он должен идти последовательно, так как это "обратное распространение ошибки"
                            //тут паралеллизовать нечего
                            for (var hiddenLayerIndex = _mlp.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                            {
                                //определяем слои
                                var prevLayer = _mlp.Layers[hiddenLayerIndex - 1];
                                var currentLayer = _mlp.Layers[hiddenLayerIndex];
                                var nextLayer = _mlp.Layers[hiddenLayerIndex + 1];

                                const uint hiddenLocalSize = 256;
                                uint hiddenGlobalSize = hiddenLocalSize*(uint) currentLayer.NonBiasNeuronCount;

                                if (inBatchIndex == 0)
                                {
                                    _hiddenKernelOverwrite[hiddenLayerIndex]
                                        .SetKernelArgMem(0, _dropConnectContainers[hiddenLayerIndex].NetMem)

                                        .SetKernelArgMem(1, _dropConnectContainers[hiddenLayerIndex - 1].StateMem)
                                        .SetKernelArgMem(2, _dropConnectContainers[hiddenLayerIndex].StateMem)
                                        .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                        .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])

                                        .SetKernelArgMem(5, _dropConnectContainers[hiddenLayerIndex].WeightMem)
                                        .SetKernelArgMem(6, _dropConnectContainers[hiddenLayerIndex + 1].WeightMem)

                                        .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])

                                        .SetKernelArgMem(8, _dropConnectPropagators[hiddenLayerIndex].MaskContainer.MaskMem)

                                        .SetKernelArg(9, 4, prevLayer.Neurons.Length)
                                        .SetKernelArg(10, 4, currentLayer.NonBiasNeuronCount)
                                        .SetKernelArg(11, 4, nextLayer.NonBiasNeuronCount)

                                        .SetKernelArg(12, 4, learningRate)
                                        .SetKernelArg(13, 4, _config.RegularizationFactor)
                                        .SetKernelArg(14, 4, (float) (data.Count))

                                        .SetKernelArg(15, 4, _dropConnectPropagators[hiddenLayerIndex].MaskContainer.BitMask)

                                        .SetKernelArgLocalMem(16, hiddenLocalSize*sizeof (float))

                                        .EnqueueNDRangeKernel(
                                            new uint[]
                                            {
                                                hiddenGlobalSize
                                            }
                                            , new uint[]
                                            {
                                                hiddenLocalSize
                                            }
                                        );
                                }
                                else
                                {
                                    _hiddenKernelIncrement[hiddenLayerIndex]
                                        .SetKernelArgMem(0, _dropConnectContainers[hiddenLayerIndex].NetMem)

                                        .SetKernelArgMem(1, _dropConnectContainers[hiddenLayerIndex - 1].StateMem)
                                        .SetKernelArgMem(2, _dropConnectContainers[hiddenLayerIndex].StateMem)
                                        .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                        .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])

                                        .SetKernelArgMem(5, _dropConnectContainers[hiddenLayerIndex].WeightMem)
                                        .SetKernelArgMem(6, _dropConnectContainers[hiddenLayerIndex + 1].WeightMem)

                                        .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])

                                        .SetKernelArgMem(8, _dropConnectPropagators[hiddenLayerIndex].MaskContainer.MaskMem)

                                        .SetKernelArg(9, 4, prevLayer.Neurons.Length)
                                        .SetKernelArg(10, 4, currentLayer.NonBiasNeuronCount)
                                        .SetKernelArg(11, 4, nextLayer.NonBiasNeuronCount)

                                        .SetKernelArg(12, 4, learningRate)
                                        .SetKernelArg(13, 4, _config.RegularizationFactor)
                                        .SetKernelArg(14, 4, (float) (data.Count))

                                        .SetKernelArg(15, 4, _dropConnectPropagators[hiddenLayerIndex].MaskContainer.BitMask)

                                        .SetKernelArgLocalMem(16, hiddenLocalSize*sizeof (float))

                                        .EnqueueNDRangeKernel(
                                            new uint[]
                                            {
                                                hiddenGlobalSize
                                            }
                                            , new uint[]
                                            {
                                                hiddenLocalSize
                                            }
                                        );
                                }
                            }

                            #endregion

                            #region logging

                            var logStep = data.Count/100;
                            if (logStep > 0 && currentIndex%logStep == 0)
                            {
                                ConsoleAmbientContext.Console.Write(
                                    "Epoche progress: {0}%, {1}      ",
                                    (currentIndex*100/data.Count),
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
                        var weightMem = _dropConnectContainers[layerIndex].WeightMem;
                        var nablaMem = _nablaWeights[layerIndex];

                        _updateWeightKernel
                            .SetKernelArgMem(0, weightMem)
                            .SetKernelArgMem(1, nablaMem)
                            .SetKernelArg(2, 4, (float)(_config.BatchSize))
                            .SetKernelArg(3, 4, weightMem.Array.Length)
                            .EnqueueNDRangeKernel(weightMem.Array.Length);
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
            foreach (var container in _dropConnectContainers)
            {
                if (container.WeightMem != null)
                {
                    container.WeightMem.Read(BlockModeEnum.Blocking);
                }
            }

            //write new weights and biases into network
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var weightLayer = _dropConnectContainers[layerIndex].WeightMem;

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
