using System;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.BackpropagationFactory.DropConnect.Bit.OpenCL.CPU
{
    /// <summary>
    /// Factory for dropconnect backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class DropConnectBitOpenCLBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
        private readonly int _sampleCount;
        private readonly float _p;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public DropConnectBitOpenCLBackpropagationAlgorithmFactory(
            int sampleCount,
            float p)
        {
            if (sampleCount <= 0)
            {
                throw new ArgumentOutOfRangeException("sampleCount");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

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
                randomizer,
                new DropConnectBitCPUBackpropagationEpocheTrainer<VectorizedLayerInference>(
                    randomizer,
                    VectorizationSizeEnum.VectorizationMode16,
                    net,
                    config,
                    clProvider,
                    _sampleCount,
                    _p),
                artifactContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
