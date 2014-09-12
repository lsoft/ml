using System;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.GPU
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables GPU-OpenCL
    /// </summary>
    public class GPUBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
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
                new GPUBackpropagationEpocheTrainer(
                    net,
                    config,
                    clProvider),
                artifactContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
