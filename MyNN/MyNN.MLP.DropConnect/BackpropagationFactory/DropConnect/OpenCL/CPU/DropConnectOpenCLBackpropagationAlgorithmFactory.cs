using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.CPU;
using MyNN.MLP.DropConnect.ForwardPropagation.Inference.OpenCL.CPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.BackpropagationFactory.DropConnect.OpenCL.CPU
{
    /// <summary>
    /// Factory for dropconnect backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class DropConnectOpenCLBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly int _sampleCount;
        private readonly float _p;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="mlpContainerHelper">Helper class for save-load MLPs</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public DropConnectOpenCLBackpropagationAlgorithmFactory(
            IMLPContainerHelper mlpContainerHelper,
            int sampleCount,
            float p)
        {
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            if (sampleCount <= 0)
            {
                throw new ArgumentOutOfRangeException("sampleCount");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _mlpContainerHelper = mlpContainerHelper;
            _sampleCount = sampleCount;
            _p = p;
        }

        public BackpropagationAlgorithm GetBackpropagationAlgorithm(
            IRandomizer randomizer,
            CLProvider clProvider,
            IArtifactContainer artifactContainer,
            IMLP net,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (net == null)
            {
                throw new ArgumentNullException("net");
            }
            if (validationDataProvider == null)
            {
                throw new ArgumentNullException("validationDataProvider");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            var algo = new BackpropagationAlgorithm(
                new DropConnectEpocheTrainer<VectorizedLayerInferencer>(
                    randomizer,
                    VectorizationSizeEnum.VectorizationMode16,
                    net,
                    config,
                    clProvider,
                    _sampleCount,
                    _p),
                _mlpContainerHelper,
                artifactContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
