using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.BackpropagationFactory.Dropout.OpenCL.GPU
{
    /// <summary>
    /// Factory for dropout backpropagation algorithm enables GPU-OpenCL
    /// </summary>
    public class GPUDropoutBackpropagationFactory : IBackpropagationFactory
    {
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly float _p;

        public GPUDropoutBackpropagationFactory(
            IMLPContainerHelper mlpContainerHelper,
            IOpenCLMaskContainerFactory maskContainerFactory,
            float p
            )
        {
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            if (maskContainerFactory == null)
            {
                throw new ArgumentNullException("maskContainerFactory");
            }

            _mlpContainerHelper = mlpContainerHelper;
            _maskContainerFactory = maskContainerFactory;
            _p = p;
        }

        public IBackpropagation CreateBackpropagation(
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

            var algo = new MLP.Backpropagation.Backpropagation(
                new GPUDropoutEpocheTrainer(
                    randomizer,
                    _maskContainerFactory,
                    net,
                    config,
                    clProvider,
                    _p
                    ),
                _mlpContainerHelper,
                artifactContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
