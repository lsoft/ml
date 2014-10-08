using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.AutoencoderMLP.OpenCL.CPU
{
    /// <summary>
    /// Regularized NLNCA backpropagation epoche trainer for autoencoders that enables CPU-OpenCL.
    /// For details please refer https://www.cs.toronto.edu/~hinton/absps/nonlinnca.pdf
    /// </summary>
    public class CPUAutoencoderNLNCAEpocheTrainer: IEpocheTrainer
    {
        private readonly IMemLayerContainer[] _containers;
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;
        private readonly Func<List<DataItem>, IDodfCalculator> _dodfCalculatorFactory;

        /// <summary>
        /// Номер слоя, на который оказывается давление NCA
        /// </summary>
        private readonly int _ncaLayerIndex;

        /// <summary>
        /// Коэффициент регуляризации NCA
        /// </summary>
        private readonly float _lambda;

        /// <summary>
        /// Количество нейронов на коротком слое на которые оказывается давление NCA
        /// </summary>
        private readonly int _takeIntoAccount;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        
        private MemFloat _dodfMem;
        private MemFloat _desiredOutput;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        private readonly ForwardPropagation.ForwardPropagation _forwardPropagation;

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
        /// <param name="mlp">Autoencoder</param>
        /// <param name="config">Learning config</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="dodfCalculatorFactory">dOdF calculator factory (for details about dOdF please refer https://www.cs.toronto.edu/~hinton/absps/nonlinnca.pdf )</param>
        /// <param name="ncaLayerIndex">Layer index to put in NCA)</param>
        /// <param name="lambda">Regularization coefficient (for details about regularization coef please refer https://www.cs.toronto.edu/~hinton/absps/nonlinnca.pdf )</param>
        /// <param name="takeIntoAccount">Neuron count on "NCA-layer" for NCA-pressure</param>
        public CPUAutoencoderNLNCAEpocheTrainer(
            VectorizationSizeEnum vse,
            IMLP mlp,
            ILearningAlgorithmConfig config,
            CLProvider clProvider,
            Func<List<DataItem>, IDodfCalculator> dodfCalculatorFactory,
            int ncaLayerIndex,
            float lambda,
            int takeIntoAccount)
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
            if (dodfCalculatorFactory == null)
            {
                throw new ArgumentNullException("dodfCalculatorFactory");
            }

            //not any activation function is allowed to correctly work under NCA-pressure
            var allowedLayerActivationFunctionList =
                new List<Type>
                {
                    typeof(LinearFunction),
                    typeof(RLUFunction),
                    typeof(DRLUFunction)
                };

            var ncaLayerFunctionType = mlp.Layers[ncaLayerIndex].LayerActivationFunction.GetType();
            if (!allowedLayerActivationFunctionList.Contains(ncaLayerFunctionType))
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Function {0} is not allowed for NLNCA layer.",
                        ncaLayerFunctionType.Name));
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;
            _dodfCalculatorFactory = dodfCalculatorFactory;
            _ncaLayerIndex = ncaLayerIndex;
            _lambda = lambda;
            _takeIntoAccount = takeIntoAccount;

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

            _forwardPropagation = new ForwardPropagation.ForwardPropagation(
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

            _dodfMem = _clProvider.CreateFloatMem(
                _mlp.Layers[_ncaLayerIndex].NonBiasNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

        }

        private void LoadPrograms()
        {
            var kg = new AutoendoderNLNCAKernelConstructor(
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
                NLNCAKernelConstructor.UpdateWeightKernelSource,
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
            if (data.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("Датасет для данного алгоритма не должен быть автоенкодерным");
            }

            if (_config.BatchSize/(float) data.Count < 0.05f)
            {
                ConsoleAmbientContext.Console.WriteWarning(
                    "Probably batch size = {0} is too low for train dataset with {1} items.",
                    _config.BatchSize,
                    data.Count);
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

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //process data set
            var currentIndex = 0;
            do
            {
                #region obtain dodf calculator

                List<DataItem> uzkii;
                List<int> uzSootv;
                
                this.ObtainUzkiiData(
                    data,
                    out uzSootv,
                    out uzkii);

                var dodfCalculator = _dodfCalculatorFactory(uzkii);

                #endregion

                //#region массово рассчитываем значения dOdF для батча

                //var inBatchDict = new Dictionary<int, float[]>();
                //for (int inBatchIndex = currentIndex, batchIndex = 0; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; ++inBatchIndex, ++batchIndex)
                //{
                //    //train data
                //    var trainData = data[inBatchIndex];

                //    if (trainData.OutputIndex >= 0)
                //    {
                //        inBatchDict.Add(inBatchIndex, new float[_dodfMem.Array.Length]);
                //    }
                //}

                //System.Threading.Tasks.Parallel.ForEach(inBatchDict, pair =>
                //{
                //    var inBatchIndex = pair.Key;

                //    var uzIndex = uzSootv[inBatchIndex];

                //    var dodf = dodfCalculator.CalculateDodf(uzIndex);
                //    dodf.CopyTo(pair.Value, 0L);
                //});

                //#endregion


                //process one batch
                for (int inBatchIndex = currentIndex, batchIndex = 0; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; ++inBatchIndex, ++batchIndex)
                {
                    //train data
                    var trainData = data[inBatchIndex];

                    //---------------------------- forward pass ----------------------------

                    _forwardPropagation.Propagate(trainData);

                    //---------------------------- backward pass, error propagation ----------------------------

                    //производная по компонентам близости (если итем с меткой)
                    if (trainData.OutputIndex >= 0)
                    {
                        var uzIndex = uzSootv[inBatchIndex];
                        var dodf = dodfCalculator.CalculateDodf(uzIndex);
                        dodf.CopyTo(_dodfMem.Array, 0);

                        //inBatchDict[inBatchIndex].CopyTo(_dodfMem.Array, 0);

                        _dodfMem.Write(BlockModeEnum.NonBlocking);
                    }

                    //отправляем на OpenCL желаемые выходы
                    trainData.Input.CopyTo(_desiredOutput.Array, 0); //инпут, так как, несмотря на то, что это автоенкодер, нам требуются и labels и датасет не автоенкодерный
                    _desiredOutput.Write(BlockModeEnum.NonBlocking);

                    //output layer
                    var outputLayerIndex = _mlp.Layers.Length - 1;

                    var outputLayer = _mlp.Layers[outputLayerIndex];
                    var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

                    //если итем с меткой, применяем NLNCA коэффициент
                    var outputLambda = 1f;
                    if (trainData.OutputIndex >= 0)
                    {
                        outputLambda = (1f - _lambda);
                    }

                    if (batchIndex == 0)
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
                            .SetKernelArg(11, 4, outputLambda)
                            .SetKernelArg(12, 4, learningRate)
                            .SetKernelArg(13, 4, _config.RegularizationFactor)
                            .SetKernelArg(14, 4, (float)(data.Count))
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
                            .SetKernelArg(11, 4, outputLambda)
                            .SetKernelArg(12, 4, learningRate)
                            .SetKernelArg(13, 4, _config.RegularizationFactor)
                            .SetKernelArg(14, 4, (float)(data.Count))
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

                        //если слой - слой для давления NCA и итем с меткой -
                        //то оказываем давление NCA
                        var hiddenLambda = 0f;
                        if (hiddenLayerIndex == _ncaLayerIndex)
                        {
                            if (trainData.OutputIndex >= 0)
                            {
                                //канает по номеру слоя и по обрабатываемому итему
                                //используем заданный коэффициент
                                hiddenLambda = _lambda;
                            }
                        }

                        if (batchIndex == 0)
                        {
                            _hiddenKernelOverwrite[hiddenLayerIndex]
                                .SetKernelArgMem(0, _containers[hiddenLayerIndex].NetMem)
                                .SetKernelArgMem(1, _containers[hiddenLayerIndex - 1].StateMem)
                                .SetKernelArgMem(2, _containers[hiddenLayerIndex].StateMem)
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _containers[hiddenLayerIndex].WeightMem)
                                .SetKernelArgMem(6, _containers[hiddenLayerIndex + 1].WeightMem)
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArgMem(8, _dodfMem)
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(11, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(12, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(14, 4, _takeIntoAccount)
                                .SetKernelArg(15, 4, hiddenLambda)
                                .SetKernelArg(16, 4, learningRate)
                                .SetKernelArg(17, 4, _config.RegularizationFactor)
                                .SetKernelArg(18, 4, (float)(data.Count))
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
                                .SetKernelArgMem(6, _containers[hiddenLayerIndex + 1].WeightMem)
                                .SetKernelArgMem(7, _nablaWeights[hiddenLayerIndex])
                                .SetKernelArgMem(8, _dodfMem)
                                .SetKernelArg(9, 4, prevLayer.Neurons.Length / 4)
                                .SetKernelArg(10, 4, prevLayer.Neurons.Length - (prevLayer.Neurons.Length % 4))
                                .SetKernelArg(11, 4, prevLayer.Neurons.Length)
                                .SetKernelArg(12, 4, currentLayer.NonBiasNeuronCount)
                                .SetKernelArg(13, 4, nextLayer.NonBiasNeuronCount)
                                .SetKernelArg(14, 4, _takeIntoAccount)
                                .SetKernelArg(15, 4, hiddenLambda)
                                .SetKernelArg(16, 4, learningRate)
                                .SetKernelArg(17, 4, _config.RegularizationFactor)
                                .SetKernelArg(18, 4, (float)(data.Count))
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


                    // Make sure we're done with everything that's been requested before
                    _clProvider.QueueFinish();
                }

                //update weights and bias into opencl memory wrappers

                for (int layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                {
                    var weightMem = _containers[layerIndex].WeightMem;
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

                #region записываем веса в сеть, чтобы следующий цикл просчета uzkii не затер веса (он выполняет PushWeights)

                //считываем веса с устройства
                PopWeights();

                //write new weights and biases into network
                WritebackWeightsToMLP();

                #endregion

                currentIndex += _config.BatchSize;
            } while (currentIndex < data.Count);

            #endregion

            ConsoleAmbientContext.Console.Write(new string(' ', 60));
            ConsoleAmbientContext.Console.ReturnCarriage();

            //конец эпохи обучения
        }

        private void ObtainUzkiiData(
            IDataSet data,
            out List<int> uzSootv,
            out List<DataItem> uzkii)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            uzSootv = new List<int>();
            uzkii = new List<DataItem>();

            var state = this._forwardPropagation.ComputeState(data);
            var output = state.ConvertAll(j => j.LState[_ncaLayerIndex]).ToList().ConvertAll(j => j.NState);

            for (var uzIndex = 0; uzIndex < data.Count; uzIndex++)
            {
                var d = data[uzIndex];

                if (d.OutputIndex >= 0)
                {
                    uzkii.Add(
                        new DataItem(
                            output[uzIndex].Take(_takeIntoAccount).ToArray(),
                            d.Output));

                    uzSootv.Add(uzkii.Count - 1);
                }
                else
                {
                    uzSootv.Add(-1);
                }
            }
        }

        private void WritebackWeightsToMLP()
        {
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

        private void PopWeights()
        {
            //считываем веса с устройства
            foreach (var container in _containers)
            {
                if (container.WeightMem != null)
                {
                    container.WeightMem.Read(BlockModeEnum.Blocking);
                }
            }
        }
    }
    
}
