using System;
using System.Collections.Generic;
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

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Transpose
{
    public class CPUTransposeBackpropagationAlgorithm : IEpocheTrainer
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

        private readonly OpenCLForwardPropagation _forwardPropagation;
        public IForwardPropagation ForwardPropagation
        {
            get
            {
                return
                    _forwardPropagation;
            }
        }

        public CPUTransposeBackpropagationAlgorithm(
            VectorizationSizeEnum vse,
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

            if (config.BatchSize == 1)
            {
                ConsoleAmbientContext.Console.WriteWarning("This backpropagation algorithm optimized to work in batch mode (typical with batch size = [25;100]). Online backpropagation is not an optimal choise. Try to use CPUTranspose2BackpropagationAlgorithm.");
            }

            if (config.BatchSize > 1 && config.BatchSize < 25)
            {
                ConsoleAmbientContext.Console.WriteWarning("Too low minibatch size ({0}) to achieve optimal performance. Try to increase batch size to 25 minimum.", config.BatchSize);
            }

            _mlp = mlp;
            _config = config;
            _clProvider = clProvider;

            _forwardPropagation = new OpenCLForwardPropagation(
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

            _transposers = new IOpenCLTransposer[_mlp.Layers.Length];
        }

        private void LoadPrograms()
        {
            var kg = new TransposeKernelConstructor(
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
                TransposeKernelConstructor.UpdateWeightKernelSource,
                "UpdateWeightKernel");
        }

        #endregion

        public void PreTrainInit(DataSet data)
        {
            //создаем инфраструктуру
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

                _transposers[i] = new TransposerNvidia(
                    _clProvider,
                    _forwardPropagation.WeightMem[i],
                    _mlp.Layers[i - 1].Neurons.Length,
                    _mlp.Layers[i].NonBiasNeuronCount);
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
                                .SetKernelArgMem(6, _transposers[hiddenLayerIndex + 1].Destination)
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

                    //обновляем веса
                    _updateWeightKernel
                        .SetKernelArgMem(0, weightMem)
                        .SetKernelArgMem(1, nablaMem)
                        .SetKernelArg(2, 4, weightMem.Array.Length)
                        .SetKernelArg(3, 4, perKernelFloats)
                        .SetKernelArg(4, 4, (float)(_config.BatchSize))
                        .EnqueueNDRangeKernel(kernelCount);

                    //транспонируем
                    _transposers[layerIndex].Transpose();
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
            foreach (var wm in _forwardPropagation.WeightMem)
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

    }
}
