using System;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
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
            IMLPContainer mlpContainer,
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
            if (mlpContainer == null)
            {
                throw new ArgumentNullException("mlpContainer");
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
                new GPUBackpropagationEpocheTrainer(
                    net,
                    config,
                    clProvider),
                mlpContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
