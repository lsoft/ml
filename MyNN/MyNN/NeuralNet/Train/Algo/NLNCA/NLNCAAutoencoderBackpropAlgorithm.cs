using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator;

namespace MyNN.NeuralNet.Train.Algo.NLNCA
{
    public class NLNCAAutoencoderBackpropAlgorithm : BaseTrainAlgorithm
    {
        private readonly Func<List<DataItem>, IDodfCalculator> _dodfCalculatorFactory;

        /// <summary>
        /// Коэффициент регуляризации NCA
        /// </summary>
        private readonly float _lambda;

        /// <summary>
        /// Количество нейронов на коротком слое на которые оказывается давление NCA
        /// </summary>
        private readonly int _takeIntoAccount;

        private readonly float[][][] _nablaWeights;
        private readonly int _ncaLayerIndex;

        public NLNCAAutoencoderBackpropAlgorithm(
            MultiLayerNeuralNetwork network,
            ILearningAlgorithmConfig config,
            MultilayerTrainProcessDelegate validation,
            Func<List<DataItem>, IDodfCalculator> dodfCalculatorFactory,
            float lambda,
            int takeIntoAccount)
            : base(network, config, validation, false)
        {
            if (dodfCalculatorFactory == null)
            {
                throw new ArgumentNullException("dodfCalculatorFactory");
            }
            _dodfCalculatorFactory = dodfCalculatorFactory;
            _lambda = lambda;
            _takeIntoAccount = takeIntoAccount;
            _nablaWeights = new float[_network.Layers.Length][][];
            _ncaLayerIndex = _network.Layers.ToList().FindIndex(k => k.NonBiasNeuronCount == _network.Layers.Min(j => j.NonBiasNeuronCount));
        }

        protected override void PreTrainInit(DataSet data)
        {
            for (int i = 0; i < _network.Layers.Length; i++)
            {
                _nablaWeights[i] = new float[_network.Layers[i].Neurons.Length][];

                for (int j = 0; j < _network.Layers[i].Neurons.Length; j++)
                {
                    _nablaWeights[i][j] = new float[_network.Layers[i].Neurons[j].Weights.Length];
                }
            }
        }

        protected override void TrainEpoche(DataSet data, string epocheRoot, float learningRate)
        {
            if (data.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("Датасет для данного алгоритма не должен быть автоенкодерным");
            }

            #region one epoche

            //process data set
            var currentIndex = 0;
            do
            {
                #region clean accumulated error for batch, for weights and biases

                //очищаем массивы
                Parallel.For(0, _network.Layers.Length, i =>
                //for (var i = 0; i < network.Layers.Length; i++)
                {
                    for (var j = 0; j < _network.Layers[i].Neurons.Length; j++)
                    {
                        Array.Clear(_nablaWeights[i][j], 0, _nablaWeights[i][j].Length);
                    }
                }
                ); //Parallel.For

                #endregion

                var uzSootv = new List<int>();
                var uzkii = new List<DataItem>();

                for (var uzIndex = 0; uzIndex < data.Count; uzIndex++)
                {
                    var d = data[uzIndex];

                    if (d.OutputIndex >= 0)
                    {
                        _network.ComputeOutput(d.Input);

                        uzkii.Add(
                            new DataItem(
                                _network.Layers[_ncaLayerIndex].LastOutput.Take(_takeIntoAccount).ToArray(),
                                d.Output));

                        uzSootv.Add(uzkii.Count - 1);
                    }
                    else
                    {
                        uzSootv.Add(-1);
                    }
                }

                var dodfCalculator = _dodfCalculatorFactory(uzkii);

                //process one batch
                //float maxdiff = 0f;
                for (var inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; ++inBatchIndex)
                {
                    //train data
                    var trainData = data[inBatchIndex];

                    //forward pass
                    var realOutput = _network.ComputeOutput(trainData.Input);

                    //производная по компонентам близости (если итем с меткой)
                    float[] dodf = null;
                    
                    if (trainData.OutputIndex >= 0)
                    {
                        var uzIndex = uzSootv[inBatchIndex];
                        dodf = dodfCalculator.CalculateDodf(uzIndex);
                    }

                    #region проверяем, что оригинальный вариант не отличается от оптимизированного

                    //var dodfold = new DodfCalculatorOld(uzkii).CalculateDodf(workIndex);

                    //for (var cc = 0; cc < dodf.Length; cc++)
                    //{
                    //    var diff = Math.Abs(dodf[cc] - dodfold[cc]);
                    //    if (maxdiff < diff)
                    //    {
                    //        maxdiff = diff;
                    //        Console.WriteLine("**************************");
                    //        Console.WriteLine(maxdiff);
                    //        Console.WriteLine("**************************");
                    //    }

                    //    if (Math.Abs(dodf[cc] - dodfold[cc]) > 1e-5)
                    //    {
                    //        throw new InvalidOperationException("<->");
                    //    }
                    //}

                    #endregion
                    //*/

                    //backward pass, error propagation
                    //last layer
                    var lastLayerIndex = _network.Layers.Length - 1;
                    var lastLayer = _network.Layers[lastLayerIndex];
                    var preLastLayerIndex = lastLayerIndex - 1;
                    var preLastLayer = _network.Layers[preLastLayerIndex];

                    for (int neuronIndex = 0,
                             neuronCount = lastLayer.Neurons.Length; neuronIndex < neuronCount; neuronIndex++)
                    {
                        var neuron = lastLayer.Neurons[neuronIndex];

                        var @out = realOutput[neuronIndex];

                        var diff = 
                            (trainData.Input[neuronIndex] - @out); //автоенкодер, поэтому trainData.Input!

                        //если итем с меткой, применяем NLNCA коэффициент
                        if (trainData.OutputIndex >= 0)
                        {
                            diff *= (1f - _lambda);
                        }

                        var nabla =
                            neuron.ActivationFunction.ComputeFirstDerivative(@out)
                            //@out * (1 - @out)
                            * diff;

                        neuron.Dedz = nabla;

                        for (int weightIndex = 0,
                                 weightCount = neuron.Weights.Length; weightIndex < weightCount; ++weightIndex)
                        {
                            var deltaWeight =
                                learningRate *
                                nabla *
                                (preLastLayer.Neurons[weightIndex].LastState
                                 + _config.RegularizationFactor *
                                 neuron.Weights[weightIndex] /
                                 data.Count);

                            _nablaWeights[lastLayerIndex][neuronIndex][weightIndex] += deltaWeight;
                        }
                    }

                    //hidden layers
                    for (int hiddenLayerIndex = _network.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
                    {
                        var prevLayer = _network.Layers[hiddenLayerIndex - 1];

                        var currentLayer = _network.Layers[hiddenLayerIndex];

                        var nextLayerIndex = hiddenLayerIndex + 1;
                        var nextLayer = _network.Layers[nextLayerIndex];

                        Parallel.For(0, currentLayer.NonBiasNeuronCount, currentNeuronIndex =>
                        //for (int currentNeuronIndex = 0, neuronCount = currentLayer.NonBiasNeuronCount; currentNeuronIndex < neuronCount; ++currentNeuronIndex)
                        {
                            var currentNeuron = currentLayer.Neurons[currentNeuronIndex];
                            var currentNablaNeuron = _nablaWeights[hiddenLayerIndex][currentNeuronIndex];

                            currentNeuron.Dedz = 0.0f;

                            for (int nextNeuronIndex = 0,
                                    nextLayerNeuronCount = nextLayer.NonBiasNeuronCount; nextNeuronIndex < nextLayerNeuronCount; nextNeuronIndex++)
                            {
                                var nextLayerNeuron = nextLayer.Neurons[nextNeuronIndex];

                                var nextWeight = nextLayerNeuron.Weights[currentNeuronIndex];
                                var nextNabla = nextLayerNeuron.Dedz;
                                var multiplied = nextWeight * nextNabla;

                                currentNeuron.Dedz += multiplied;
                            }

                            //если слой - самый узкий, если обрабатываемый нейрон входит в число тех,
                            //на которые оказывается давление NCA, и итем с меткой -
                            //то оказываем давление NCA
                            if (hiddenLayerIndex == _ncaLayerIndex)
                            {
                                if (currentNeuronIndex < _takeIntoAccount)
                                {
                                    if (trainData.OutputIndex >= 0)
                                    {
                                        currentNeuron.Dedz += _lambda * dodf[currentNeuronIndex];
                                    }
                                }
                            }

                            var @out = currentNeuron.LastState;
                            currentNeuron.Dedz *=
                                currentNeuron.ActivationFunction.ComputeFirstDerivative(@out);

                            for (int currentWeightIndex = 0,
                                    weightCount = currentNeuron.Weights.Length; currentWeightIndex < weightCount; ++currentWeightIndex)
                            {
                                var prevOut =
                                    prevLayer.Neurons[currentWeightIndex].LastState;

                                currentNablaNeuron[currentWeightIndex] +=
                                    learningRate *
                                    currentNeuron.Dedz *
                                    (prevOut + _config.RegularizationFactor * currentNeuron.Weights[currentWeightIndex] / data.Count);
                            }
                        }
                        ); //Parallel.For
                    }

                    int logStep = data.Count / 100;
                    if (logStep > 0 && currentIndex % logStep == 0)
                    {
                        Console.Write("Epoche progress: " + (currentIndex * 100 / data.Count) + "%, " + DateTime.Now.ToString() + "      ");
                        Console.SetCursorPosition(0, Console.CursorTop);
                    }
                }

                //update weights and bias
                //Parallel.For(1, network.Layers.Length, layerIndex =>
                for (int layerIndex = 1; layerIndex < _network.Layers.Length; ++layerIndex)
                {
                    var layer = _network.Layers[layerIndex];
                    var nablaLayer = _nablaWeights[layerIndex];

                    for (int neuronIndex = 0,
                             neuronCount = layer.NonBiasNeuronCount; neuronIndex < neuronCount; ++neuronIndex)
                    {
                        var neuron = layer.Neurons[neuronIndex];
                        var nablaNeuron = nablaLayer[neuronIndex];

                        for (int weightIndex = 0,
                                 weightCount = neuron.Weights.Length; weightIndex < weightCount; ++weightIndex)
                        {
                            neuron.Weights[weightIndex] += nablaNeuron[weightIndex] / _config.BatchSize;
                        }
                    }
                }
                //); //Parallel.For

                currentIndex += _config.BatchSize;
            } while (currentIndex < data.Count);

            #endregion

            ////постим информацию о значениях внутри слоя NLNCA
            //DebugUzkiiInfo(data);
        }

        private void DebugUzkiiInfo(DataSet data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var uzkii = new List<DataItem>();

            for (var uzIndex = 0; uzIndex < data.Count; uzIndex++)
            {
                var d = data[uzIndex];

                if (d.OutputIndex >= 0)
                {
                    _network.ComputeOutput(d.Input);

                    uzkii.Add(
                        new DataItem(
                            _network.Layers[_ncaLayerIndex].LastOutput.Take(_takeIntoAccount).ToArray(),
                            d.Output));
                }
            }

            if (uzkii.Count > 0)
            {
                var dimensionCount = uzkii[0].Input.Length;
                for (var d = 0; d < dimensionCount; d++)
                {
                    var min = uzkii.Min(j => j.Input[d]);
                    var max = uzkii.Max(j => j.Input[d]);

                    Console.WriteLine(
                        "Dimension {0} contains min={1}, max={2}",
                        d,
                        min,
                        max);
                }
            }
        }

    }
}

