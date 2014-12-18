using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp
{
    /// <summary>
    /// Classic backpropagation epoche trainer
    /// </summary>
    public class CSharpEpocheTrainer : IEpocheTrainer
    {
        private readonly ICSharpLayerContainer[] _containers;
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private float[][] _deDz;
        private float[][] _nablaWeights;
        private float[] _desiredOutput;

        private HiddenLayerKernel[] _hiddenKernels;
        private OutputLayerKernel _outputKernel;
        private UpdateWeightKernel _updateWeightKernel;

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
        public CSharpEpocheTrainer(
            IMLP mlp,
            ILearningAlgorithmConfig config
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
            _mlp = mlp;
            _config = config;


            var cc = new CSharpPropagatorComponentConstructor(
                );

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            cc.CreateComponents(
                mlp,
                out containers,
                out propagators);

            _containers = containers.ToList().ConvertAll(j => j as ICSharpLayerContainer).ToArray();

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
            _nablaWeights = new float[_mlp.Layers.Length][];
            _deDz = new float[_mlp.Layers.Length][];
        }

        private void LoadPrograms()
        {
            _hiddenKernels = new HiddenLayerKernel[_mlp.Layers.Length];

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                _hiddenKernels[layerIndex] = new HiddenLayerKernel(
                    _mlp.Layers[layerIndex],
                    _config
                    );
            }

            _outputKernel = new OutputLayerKernel(
                _mlp.Layers.Last(),
                _config
                );

            //определяем кернел обновления весов
            _updateWeightKernel = new UpdateWeightKernel();
        }

        #endregion

        public void PreTrainInit(IDataSet data)
        {
            //создаем массивы смещений по весам и dedz
            for (var i = 1; i < _mlp.Layers.Length; i++)
            {
                var lastLayer = i == (_mlp.Layers.Length - 1);
                var biasNeuronCount = lastLayer ? 0 : 1;

                _nablaWeights[i] = new float[
                    (_mlp.Layers[i].Neurons.Length - biasNeuronCount)*_mlp.Layers[i].Neurons[0].Weights.Length
                    ];

                _deDz[i] = new float[_mlp.Layers[i].NonBiasNeuronCount];
            }

            var outputLength = _mlp.Layers.Last().NonBiasNeuronCount;

            //создаем объекты желаемых выходных данных (т.е. правильных ответов сети)
            _desiredOutput = new float[outputLength];
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

            _forwardPropagation.ClearAndPushHiddenLayers();

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
                            trainDataItem.Output.CopyTo(_desiredOutput, 0);

                            #region output layer

                            var outputLayerIndex = _mlp.Layers.Length - 1;

                            var outputLayer = _mlp.Layers[outputLayerIndex];
                            var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                            var outputNablaLayer = _nablaWeights[outputLayerIndex];

                            if (inBatchIndex == 0)
                            {
                                _outputKernel.CalculateOverwrite(
                                    _containers[outputLayerIndex].NetMem,
                                    _containers[outputLayerIndex - 1].StateMem,
                                    _containers[outputLayerIndex].StateMem,
                                    this._deDz[outputLayerIndex],
                                    _desiredOutput,
                                    _containers[outputLayerIndex].WeightMem,
                                    outputNablaLayer,
                                    preOutputLayer.Neurons.Length,
                                    outputLayer.NonBiasNeuronCount,
                                    learningRate,
                                    _config.RegularizationFactor,
                                    (float) (data.Count)
                                    );

                            }
                            else
                            {
                                _outputKernel.CalculateIncrement(
                                    _containers[outputLayerIndex].NetMem,
                                    _containers[outputLayerIndex - 1].StateMem,
                                    _containers[outputLayerIndex].StateMem,
                                    this._deDz[outputLayerIndex],
                                    _desiredOutput,
                                    _containers[outputLayerIndex].WeightMem,
                                    outputNablaLayer,
                                    preOutputLayer.Neurons.Length,
                                    outputLayer.NonBiasNeuronCount,
                                    learningRate,
                                    _config.RegularizationFactor,
                                    (float)(data.Count)
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

                                if (inBatchIndex == 0)
                                {
                                    _hiddenKernels[hiddenLayerIndex].CalculateOverwrite(
                                        _containers[hiddenLayerIndex].NetMem,
                                        _containers[hiddenLayerIndex - 1].StateMem,
                                        _containers[hiddenLayerIndex].StateMem,
                                        this._deDz[hiddenLayerIndex],
                                        this._deDz[hiddenLayerIndex + 1],
                                        _containers[hiddenLayerIndex].WeightMem,
                                        _containers[hiddenLayerIndex + 1].WeightMem,
                                        _nablaWeights[hiddenLayerIndex],
                                        prevLayer.Neurons.Length,
                                        currentLayer.NonBiasNeuronCount,
                                        nextLayer.NonBiasNeuronCount,
                                        learningRate,
                                        _config.RegularizationFactor,
                                        (float) (data.Count)
                                        );
                                }
                                else
                                {
                                    _hiddenKernels[hiddenLayerIndex].CalculateIncrement(
                                        _containers[hiddenLayerIndex].NetMem,
                                        _containers[hiddenLayerIndex - 1].StateMem,
                                        _containers[hiddenLayerIndex].StateMem,
                                        this._deDz[hiddenLayerIndex],
                                        this._deDz[hiddenLayerIndex + 1],
                                        _containers[hiddenLayerIndex].WeightMem,
                                        _containers[hiddenLayerIndex + 1].WeightMem,
                                        _nablaWeights[hiddenLayerIndex],
                                        prevLayer.Neurons.Length,
                                        currentLayer.NonBiasNeuronCount,
                                        nextLayer.NonBiasNeuronCount,
                                        learningRate,
                                        _config.RegularizationFactor,
                                        (float)(data.Count)
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
                                    ((long)currentIndex * 100 / data.Count),
                                    DateTime.Now.ToString());

                                ConsoleAmbientContext.Console.ReturnCarriage();
                            }

                            #endregion

                            #endregion
                        }
                    }

                    #region update weights and bias

                    for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                    {
                        var weightMem = _containers[layerIndex].WeightMem;
                        var nablaMem = _nablaWeights[layerIndex];

                        _updateWeightKernel.UpdateWeigths(
                            weightMem,
                            nablaMem,
                            (float)_config.BatchSize
                            );
                    }

                    #endregion

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

            //write new weights and biases into network
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var weightLayer = _containers[layerIndex].WeightMem;

                var weightShiftIndex = 0;
                for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; ++neuronIndex)
                {
                    var neuron = layer.Neurons[neuronIndex];

                    var weightCount = neuron.Weights.Length;

                    Array.Copy(weightLayer, weightShiftIndex, neuron.Weights, 0, weightCount);
                    weightShiftIndex += weightCount;
                }
            }

        }

    }
}
