using System;
using System.IO;
using System.Linq;
using MyNN.Common;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Backpropagation.EpocheTrainer
{
    /// <summary>
    /// Backpropagation epoche trainer
    /// </summary>
    public class EpocheTrainer : IEpocheTrainer
    {
        private readonly ILayerContainer[] _containers;
        private readonly IMLP _mlp;
        private readonly ILearningAlgorithmConfig _config;

        private readonly ILayerBackpropagator[] _backpropagators;
        private readonly Action _batchAwaiter;

        private readonly IForwardPropagation _forwardPropagation;
        private readonly IDesiredValuesContainer _desiredValuesContainer;


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="config">Learning config</param>
        /// <param name="desiredValuesContainer">Container for targets values</param>
        /// <param name="containers"></param>
        /// <param name="backpropagators"></param>
        /// <param name="batchAwaiter">Action for wait for batch calculation finished</param>
        /// <param name="forwardPropagation"></param>
        public EpocheTrainer(
            IMLP mlp,
            ILearningAlgorithmConfig config,
            ILayerContainer[] containers,
            IDesiredValuesContainer desiredValuesContainer,
            ILayerBackpropagator[] backpropagators,
            Action batchAwaiter,
            IForwardPropagation forwardPropagation
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
            if (containers == null)
            {
                throw new ArgumentNullException("containers");
            }
            if (backpropagators == null)
            {
                throw new ArgumentNullException("backpropagators");
            }
            if (batchAwaiter == null)
            {
                throw new ArgumentNullException("batchAwaiter");
            }
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }

            _mlp = mlp;
            _config = config;
            _desiredValuesContainer = desiredValuesContainer;
            _backpropagators = backpropagators;
            _batchAwaiter = batchAwaiter;
            _forwardPropagation = forwardPropagation;
            _containers = containers;

        }

        public void PreTrainInit(IDataSet data)
        {
            //nothing to do
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

            //готовимся к эпохе
            _forwardPropagation.ClearAndPushHiddenLayers();
            _backpropagators.Foreach(j =>
            {
                if (j != null)
                {
                    j.Prepare();
                }
            });

            //process data set
            using(var enumerator = data.StartIterate())
            {
                var allowedToContinue = true;
                for (
                    var currentIndex = 0;
                    allowedToContinue;
                    currentIndex += _config.BatchSize
                    )
                {
                    var batchProcessedOneItemAtLeast = false;

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
                            var firstItemInBatch = inBatchIndex == 0;

                            //train data
                            var trainDataItem = enumerator.Current;

                            #region forward pass

                            _forwardPropagation.Propagate(trainDataItem);

                            #endregion

                            #region backward pass, error propagation

                            _desiredValuesContainer.SetValues(trainDataItem.Output);

                            for (var layerIndex = _mlp.Layers.Length - 1; layerIndex > 0; layerIndex--)
                            {
                                var backpropagator = _backpropagators[layerIndex];

                                backpropagator.Backpropagate(
                                    data.Count,
                                    learningRate,
                                    firstItemInBatch
                                    );
                            }

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

                            batchProcessedOneItemAtLeast = true;
                        }
                    }

                    #region update weights and bias into opencl memory wrappers

                    if (batchProcessedOneItemAtLeast)
                    {
                        for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
                        {
                            _backpropagators[layerIndex].UpdateWeights();
                        }
                    }

                    #endregion

                    // Make sure we're done with everything that's been requested before
                    _batchAwaiter();
                }
            }

            #endregion

            ConsoleAmbientContext.Console.Write(new string(' ', 60));
            ConsoleAmbientContext.Console.ReturnCarriage();

            //конец эпохи обучения

            //считываем веса с устройства
            foreach (var container in _containers)
            {
                if (container != null)
                {
                    container.PopWeights();
                }
            }

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                var container = _containers[layerIndex];

                container.WritebackWeightsToMLP(layer);
            }

        }
    }
}
