using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;

namespace MyNN.NeuralNet.Train.Algo
{
    public class NaiveBackpropagationLearningAlgorithm : BaseTrainAlgorithm
    {
        private readonly float[][][] _nablaWeights;

        public NaiveBackpropagationLearningAlgorithm(
            MultiLayerNeuralNetwork network,
            ILearningAlgorithmConfig config,
            MultilayerTrainProcessDelegate validation)
            : base(network, config, validation)
        {
            _nablaWeights = new float[_network.Layers.Length][][];
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

                //!!! изучить L2 регул€ризацию и проверить реализацию ее в алгоритме Naive

                //process one batch
                for (var inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; ++inBatchIndex)
                {
                    //train data
                    var trainData = data[inBatchIndex];

                    //forward pass
                    var realOutput = _network.ComputeOutput(trainData.Input);


                    //var output = new StringBuilder(1000000);
                    //for (int index = 0, upLimit = realOutput.Length; index < upLimit; ++index)
                    //{
                    //    output.AppendLine(
                    //        deleteme.ToInt(realOutput[index]).ToString());
                    //}
                    //System.IO.File.AppendAllText("_output naive.txt", output.ToString());


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

                        var nabla =
                            neuron.ActivationFunction.ComputeFirstDerivative(@out)
                            //@out * (1 - @out)
                            *(trainData.Output[neuronIndex] - @out);

                        neuron.Dedz = nabla;

                        for (int weightIndex = 0,
                                 weightCount = neuron.Weights.Length; weightIndex < weightCount; ++weightIndex)
                        {
                            var deltaWeight =
                                learningRate *
                                nabla*
                                (preLastLayer.Neurons[weightIndex].LastState
                                 + _config.RegularizationFactor*
                                 neuron.Weights[weightIndex]/
                                 data.Count);

                            _nablaWeights[lastLayerIndex][neuronIndex][weightIndex] += deltaWeight;
                        }
                    }

                    //var lasts = new StringBuilder(1000000);
                    //for (int neuronIndex = 0, neuronCount = lastLayer.Neurons.Length; neuronIndex < neuronCount; ++neuronIndex)
                    //{
                    //    for (int weightIndex = 0, weightCount = lastLayer.Neurons[neuronIndex].Weights.Length; weightIndex < weightCount; ++weightIndex)
                    //    {
                    //        lasts.AppendLine(
                    //            deleteme.ToInt(nablaWeights[lastLayerIndex][neuronIndex][weightIndex]).ToString());
                    //    }
                    //}
                    //System.IO.File.AppendAllText("_outputweight naive.txt", lasts.ToString());


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
                                var multiplied = nextWeight*nextNabla;

                                currentNeuron.Dedz += multiplied;
                            }

                            var @out = currentNeuron.LastState;
                            currentNeuron.Dedz *=
                                currentNeuron.ActivationFunction.ComputeFirstDerivative(@out);
                                //@out * (1 - @out);

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

                //var nablazz = new StringBuilder(1000000);
                //for (int neuronIndex = 0, neuronCount = nablaWeights[1].Length; neuronIndex < neuronCount; ++neuronIndex)
                //{
                //    for (int weightIndex = 0, weightCount = nablaWeights[1][neuronIndex].Length; weightIndex < weightCount; ++weightIndex)
                //    {
                //        nablazz.AppendLine(
                //            deleteme.ToInt(nablaWeights[1][neuronIndex][weightIndex]).ToString());
                //    }
                //}
                //System.IO.File.AppendAllText("ZZZ c# middle nabla.txt", nablazz.ToString());

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
                            neuron.Weights[weightIndex] += nablaNeuron[weightIndex]/_config.BatchSize;
                        }
                    }
                }
                //); //Parallel.For

                currentIndex += _config.BatchSize;
            }
            while (currentIndex < data.Count);

            #endregion
        }
    }
}

