using System;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables CPU-OpenCL with transposed weights
    /// </summary>
    public class CPUTransposeBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
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
                new CPUTransposeBackpropagationEpocheTrainer(
                    VectorizationSizeEnum.VectorizationMode16,
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
