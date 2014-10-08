using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.CPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.BackpropagationFactory.TransposedClassic.OpenCL.CPU
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables CPU-OpenCL with transposed weights
    /// </summary>
    public class CPUTransposeBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public CPUTransposeBackpropagationAlgorithmFactory(
            IMLPContainerHelper mlpContainerHelper
            )
        {
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            _mlpContainerHelper = mlpContainerHelper;
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
                new CPUTransposeEpocheTrainer(
                    VectorizationSizeEnum.VectorizationMode16,
                    net,
                    config,
                    clProvider),
                _mlpContainerHelper,
                artifactContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
