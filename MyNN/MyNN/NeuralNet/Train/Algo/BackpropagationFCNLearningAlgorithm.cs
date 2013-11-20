//using System;
//using System.Collections.Generic;
//using MyNN.NeuralNet.Structure;
//using MyNN.NeuralNet.Structure.Layers;
//using MyNN.NeuralNet.Structure.Neurons;

//namespace MyNN.NeuralNet.Train.Algo
//{
//    public class BackpropagationFCNLearningAlgorithm
//    {

//        private LearningAlgorithmConfig _config = null;
//        private readonly MultilayerTrainProcessDelegate _trainDelegate;
//        private Random _random = null;

//        public BackpropagationFCNLearningAlgorithm(
//            LearningAlgorithmConfig config,
//            MultilayerTrainProcessDelegate trainDelegate,
//            int randomSeed)
//        {
//            _config = config;
//            _trainDelegate = trainDelegate;
//            _random = new Random(randomSeed);
//        }





//        public void Train(
//            ITrainMultiLayerNetwork<ITrainLayer<ITrainableNeuron>> network,
//            IList<DataItem<double>> data)
//        {
//            if (_config.BatchSize < 1 || _config.BatchSize > data.Count)
//            {
//                _config.BatchSize = data.Count;
//            }
//            double currentError = Single.MaxValue;
//            double lastError = 0;
//            int epochNumber = 0;

//            Console.WriteLine("Start training...");

//            do
//            {
//                lastError = currentError;
//                DateTime dtStart = DateTime.Now;

//                #region one epoche

//                //preparation for epoche
//                int[] trainingIndices = new int[data.Count];
//                for (int i = 0; i < data.Count; i++)
//                {
//                    trainingIndices[i] = i;
//                }
//                if (_config.BatchSize > 0)
//                {
//                    trainingIndices = Shuffle(trainingIndices);
//                }





//                //process data set
//                int currentIndex = 0;
//                do
//                {

//                    #region initialize accumulated error for batch, for weights and biases

//                    double[][][] nablaWeights = new double[network.Layers.Length][][];
//                    //double[][] nablaBiases = new double[network.Layers.Length][];

//                    for (int i = 0; i < network.Layers.Length; i++)
//                    {
//                        //nablaBiases[i] = new double[network.Layers[i].Neurons.Length];
//                        nablaWeights[i] = new double[network.Layers[i].Neurons.Length][];
//                        for (int j = 0; j < network.Layers[i].Neurons.Length; j++)
//                        {
//                            //nablaBiases[i][j] = 0;
//                            nablaWeights[i][j] = new double[network.Layers[i].Neurons[j].Weights.Length];
//                            for (int k = 0; k < network.Layers[i].Neurons[j].Weights.Length; k++)
//                            {
//                                nablaWeights[i][j][k] = 0;
//                            }
//                        }
//                    }

//                    #endregion

//                    //process one batch
//                    for (int inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; inBatchIndex++)
//                    {
//                        //forward pass
//                        double[] realOutput = network.ComputeOutput(data[trainingIndices[inBatchIndex]].Input);

//                        //backward pass, error propagation
//                        //last layer
//                        for (int j = 0; j < network.Layers[network.Layers.Length - 1].Neurons.Length; j++)
//                        {
//                            network.Layers[network.Layers.Length - 1].Neurons[j].Dedz =
//                                _config.ErrorFunction.CalculatePartialDerivaitveByV2Index(data[inBatchIndex].Output,
//                                                                                            realOutput, j) *
//                                network.Layers[network.Layers.Length - 1].Neurons[j].ActivationFunction.
//                                    ComputeFirstDerivative(network.Layers[network.Layers.Length - 1].Neurons[j].LastNET);

//                            //nablaBiases[network.Layers.Length - 1][j] += _config.LearningRate *
//                            //                                            network.Layers[network.Layers.Length - 1].Neurons[j].Nabla;

//                            for (int i = 0; i < network.Layers[network.Layers.Length - 1].Neurons[j].Weights.Length; i++)
//                            {
//                                nablaWeights[network.Layers.Length - 1][j][i] +=
//                                    _config.LearningRate * (network.Layers[network.Layers.Length - 1].Neurons[j].Dedz *
//                                                            (network.Layers.Length > 1 ?
//                                                                network.Layers[network.Layers.Length - 1 - 1].Neurons[i].LastState :
//                                                                data[inBatchIndex].Input[i])
//                                                                +
//                                                            _config.RegularizationFactor *
//                                                            network.Layers[network.Layers.Length - 1].Neurons[j].Weights[i]
//                                                                / data.Count);
//                            }
//                        }

//                        //var lasts = new StringBuilder(1000000);
//                        //for (int neuronIndex = 0, neuronCount = network.Layers[network.Layers.Length - 1].Neurons.Length; neuronIndex < neuronCount; ++neuronIndex)
//                        //{
//                        //    for (int weightIndex = 0, weightCount = network.Layers[network.Layers.Length - 1].Neurons[neuronIndex].Weights.Length; weightIndex < weightCount; ++weightIndex)
//                        //    {
//                        //        lasts.AppendLine(
//                        //            (deleteme.ToInt(-nablaWeights[network.Layers.Length - 1][neuronIndex][weightIndex])).ToString());
//                        //    }
//                        //}
//                        //System.IO.File.AppendAllText("ZZZ fcn lasts.txt", lasts.ToString());


//                        //hidden layers
//                        for (int hiddenLayerIndex = network.Layers.Length - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
//                        {
//                            for (int j = 0; j < network.Layers[hiddenLayerIndex].NonBiasNeuronCount; j++)
//                            {
//                                network.Layers[hiddenLayerIndex].Neurons[j].Dedz = 0;
//                                for (int k = 0; k < network.Layers[hiddenLayerIndex + 1].NonBiasNeuronCount; k++)
//                                {
//                                    network.Layers[hiddenLayerIndex].Neurons[j].Dedz +=
//                                        network.Layers[hiddenLayerIndex + 1].Neurons[k].Weights[j] *
//                                        network.Layers[hiddenLayerIndex + 1].Neurons[k].Dedz;
//                                }
//                                network.Layers[hiddenLayerIndex].Neurons[j].Dedz *=
//                                    network.Layers[hiddenLayerIndex].Neurons[j].ActivationFunction.
//                                        ComputeFirstDerivative(
//                                            network.Layers[hiddenLayerIndex].Neurons[j].LastNET
//                                        );

//                                //nablaBiases[hiddenLayerIndex][j] += _config.LearningRate *
//                                //                                    network.Layers[hiddenLayerIndex].Neurons[j].Nabla;

//                                for (int i = 0; i < network.Layers[hiddenLayerIndex].Neurons[j].Weights.Length; i++)
//                                {
//                                    nablaWeights[hiddenLayerIndex][j][i] += _config.LearningRate * (
//                                        network.Layers[hiddenLayerIndex].Neurons[j].Dedz *
//                                        (hiddenLayerIndex > 0 ? network.Layers[hiddenLayerIndex - 1].Neurons[i].LastState : data[inBatchIndex].Input[i])
//                                            +
//                                        _config.RegularizationFactor * network.Layers[hiddenLayerIndex].Neurons[j].Weights[i] / data.Count
//                                        );
//                                }

//                            }
//                        }
//                    }

//                    //update weights and bias
//                    for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
//                    {
//                        for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
//                        {
//                            //network.Layers[layerIndex].Neurons[neuronIndex].Bias -= nablaBiases[layerIndex][neuronIndex];
//                            for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
//                            {
//                                network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] -=
//                                    nablaWeights[layerIndex][neuronIndex][weightIndex];
//                            }
//                        }
//                    }

//                    //var weights = new StringBuilder(1000000);
//                    //var biases = new StringBuilder(1000000);
//                    //for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
//                    //{
//                    //    for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
//                    //    {
//                    //        for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
//                    //        {
//                    //            weights.AppendLine(deleteme.ToInt(network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex]).ToString());
//                    //            biases.AppendLine(deleteme.ToInt(nablaWeights[layerIndex][neuronIndex][weightIndex]).ToString());
//                    //        }
//                    //    }
//                    //}
//                    //System.IO.File.AppendAllText("ZZZ fcn weight.txt", weights.ToString());
//                    //System.IO.File.AppendAllText("ZZZ fcn biases.txt", biases.ToString());


//                    currentIndex += _config.BatchSize;
//                } while (currentIndex < data.Count);

//                #endregion

//                //recalculating error on all data
//                //real error
//                currentError = 0;
//                for (int i = 0; i < data.Count; i++)
//                {
//                    double[] realOutput = network.ComputeOutput(data[i].Input);
//                    currentError += _config.ErrorFunction.Calculate(data[i].Output, realOutput);
//                }
//                currentError *= 1d / data.Count;
//                //regularization term
//                if (Math.Abs(_config.RegularizationFactor - 0d) > Double.Epsilon)
//                {
//                    double reg = 0;
//                    for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
//                    {
//                        for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
//                        {
//                            for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
//                            {
//                                reg += network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] *
//                                        network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex];
//                            }
//                        }
//                    }
//                    currentError += _config.RegularizationFactor * reg / (2 * data.Count);
//                }

//                epochNumber++;
//                Console.WriteLine("Epoch #" + epochNumber +
//                                    " error is " + currentError +
//                                    "; it takes: " +
//                                    (int)((DateTime.Now - dtStart).TotalMilliseconds));

//                //внешн€€ функци€ дл€ обсчета на тестовом множестве
//                //√лубокое и тормозное (но простое в написании) клонирование
//                var clonedNet = SerializationHelper.DeepClone(network);
//                _trainDelegate(clonedNet, true);

//            } while (epochNumber < _config.MaxEpoches &&
//                     currentError > _config.MinError &&
//                     Math.Abs(currentError - lastError) > _config.MinErrorChange);
//        }

//        private int[] Shuffle(int[] arr)
//        {
//            //return arr;//!!!
//            for (int i = 0; i < arr.Length - 1; i++)
//            {
//                if (_random.NextDouble() >= 0.5d)
//                {
//                    int newIndex = _random.Next(arr.Length);
//                    int tmp = arr[i];
//                    arr[i] = arr[newIndex];
//                    arr[newIndex] = tmp;
//                }
//            }
//            return arr;
//        }

//    }
//}
