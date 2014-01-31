using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.Autoencoders.BackpropagationFactory
{
    public class DropConnectBitOpenCLBackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
        private readonly int _sampleCount;
        private readonly float _p;

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
            MLP net,
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
                (processedMLP, processedConfig) =>
                    new DropConnectBitOpenCLBackpropagationAlgorithm<OpenCLLayerInferenceNew16>(
                        randomizer,
                        VectorizationSizeEnum.VectorizationMode16,
                        net,
                        config,
                        clProvider,
                        _sampleCount,
                        _p),
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
