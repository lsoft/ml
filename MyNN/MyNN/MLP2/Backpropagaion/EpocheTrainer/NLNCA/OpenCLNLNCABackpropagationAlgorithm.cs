using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA
{
    public class OpenCLNLNCABackpropagationAlgorithm: IEpocheTrainer
    {
        private readonly MLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly CLProvider _clProvider;
        private readonly Func<List<DataItem>, IDodfCalculator> _dodfCalculatorFactory;

        private MemFloat[] _deDz;
        private MemFloat[] _nablaWeights;
        private MemFloat _desiredOutput;

        private Kernel[] _hiddenKernelIncrement, _hiddenKernelOverwrite;
        private Kernel[] _outputKernelIncrement, _outputKernelOverwrite;
        private Kernel _updateWeightKernel;

        private readonly CPUForwardPropagation _forwardPropagation;
        public IForwardPropagation ForwardPropagation
        {
            get
            {
                return
                    _forwardPropagation;
            }
        }

        public OpenCLNLNCABackpropagationAlgorithm(
            VectorizationSizeEnum vse,
            MLP mlp,
            ILearningAlgorithmConfig config,
            CLProvider clProvider,
            Func<List<DataItem>, IDodfCalculator> dodfCalculatorFactory
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
            if (dodfCalculatorFactory == null)
            {
                throw new ArgumentNullException("dodfCalculatorFactory");
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;
            _dodfCalculatorFactory = dodfCalculatorFactory;

            var allowedLayerActivationFunctionList = new List<Type>
            {
                typeof(LinearFunction),
                typeof(RLUFunction)
            };

            var ncaLayerFunctionType = _mlp.Layers.Last().LayerActivationFunction.GetType();
            if (!allowedLayerActivationFunctionList.Contains(ncaLayerFunctionType))
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Слой с давлением NCA должен не должен иметь функцию {0}",
                        ncaLayerFunctionType.Name));
            }

            _forwardPropagation = new CPUForwardPropagation(
                vse,
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
        }

        private void LoadPrograms()
        {
            var kg = new NLNCAKernelConstructor(
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

        public void PreTrainInit(DataSet data)
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

                var nlncaOutput = _forwardPropagation.ComputeOutput(data);

                var uzkii = new List<DataItem>();
                var uzkiiIndex = 0;
                foreach (var d in nlncaOutput)
                {
                    uzkii.Add(
                        new DataItem(d.State, data[uzkiiIndex].Output));

                    uzkiiIndex++;
                }

                var dodfCalculator = _dodfCalculatorFactory(uzkii);

                #endregion


                //process one batch
                for (int inBatchIndex = currentIndex, batchIndex = 0; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; ++inBatchIndex, ++batchIndex)
                {
                    //train data
                    var trainData = data[inBatchIndex];

                    //---------------------------- forward pass ----------------------------

                    _forwardPropagation.Propagate(trainData);

                    //---------------------------- backward pass, error propagation ----------------------------

                    //производная по компонентам близости
                    var dodf = dodfCalculator.CalculateDodf(inBatchIndex);

                    //формируем желаемые выводы
                    dodf.CopyTo(_desiredOutput.Array, 0);

                    //отправляем на OpenCL желаемые выходы
                    _desiredOutput.Write(BlockModeEnum.Blocking);

                    //output layer
                    var outputLayerIndex = _mlp.Layers.Length - 1;

                    var outputLayer = _mlp.Layers[outputLayerIndex];
                    var preOutputLayer = _mlp.Layers[outputLayerIndex - 1];

                    var outputNablaLayer = _nablaWeights[outputLayerIndex];

                    if (batchIndex == 0)
                    {
                        _outputKernelOverwrite.Last()
                            .SetKernelArgMem(0, _forwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _forwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _forwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _forwardPropagation.WeightMem[outputLayerIndex])
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
                            .SetKernelArgMem(0, _forwardPropagation.NetMem[outputLayerIndex])
                            .SetKernelArgMem(1, _forwardPropagation.StateMem[outputLayerIndex - 1])
                            .SetKernelArgMem(2, _forwardPropagation.StateMem[outputLayerIndex])
                            .SetKernelArgMem(3, this._deDz[outputLayerIndex])
                            .SetKernelArgMem(4, _desiredOutput)
                            .SetKernelArgMem(5, _forwardPropagation.WeightMem[outputLayerIndex])
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

                        if (batchIndex == 0)
                        {
                            _hiddenKernelOverwrite[hiddenLayerIndex]
                                .SetKernelArgMem(0, _forwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _forwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _forwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _forwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _forwardPropagation.WeightMem[hiddenLayerIndex + 1])
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
                                .SetKernelArgMem(0, _forwardPropagation.NetMem[hiddenLayerIndex])
                                .SetKernelArgMem(1, _forwardPropagation.StateMem[hiddenLayerIndex - 1])
                                .SetKernelArgMem(2, _forwardPropagation.StateMem[hiddenLayerIndex])
                                .SetKernelArgMem(3, this._deDz[hiddenLayerIndex])
                                .SetKernelArgMem(4, this._deDz[hiddenLayerIndex + 1])
                                .SetKernelArgMem(5, _forwardPropagation.WeightMem[hiddenLayerIndex])
                                .SetKernelArgMem(6, _forwardPropagation.WeightMem[hiddenLayerIndex + 1])
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
                    var weightMem = _forwardPropagation.WeightMem[layerIndex];
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


                #region записываем веса в весь, чтобы следующий цикл просчета uzkii не затер веса (он выполняет PushWeights)

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

        private void WritebackWeightsToMLP()
        {
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

        private void PopWeights()
        {
            //считываем веса с устройства
            foreach (var wm in _forwardPropagation.WeightMem)
            {
                if (wm != null)
                {
                    wm.Read(BlockModeEnum.Blocking);
                }
            }
        }
    }
    
}
