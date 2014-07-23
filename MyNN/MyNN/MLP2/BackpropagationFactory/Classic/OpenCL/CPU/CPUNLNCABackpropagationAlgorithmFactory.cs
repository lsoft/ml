using System;
using System.Collections.Generic;
using MyNN.Data;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.AutoencoderMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU
{
    /// <summary>
    /// Factory for NLNCA-backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class CPUNLNCABackpropagationAlgorithmFactory : IBackpropagationAlgorithmFactory
    {
        private readonly Func<List<DataItem>, IDodfCalculator> _dodfCalculatorFactory;
        private readonly int _ncaLayerIndex;
        private readonly float _lambda;
        private readonly float _partOfTakeIntoAccount;

        public CPUNLNCABackpropagationAlgorithmFactory(
            Func<List<DataItem>, IDodfCalculator> dodfCalculatorFactory,
            int ncaLayerIndex,
            float lambda,
            float partOfTakeIntoAccount)
        {
            if (dodfCalculatorFactory == null)
            {
                throw new ArgumentNullException("dodfCalculatorFactory");
            }

            _dodfCalculatorFactory = dodfCalculatorFactory;
            _ncaLayerIndex = ncaLayerIndex;
            _lambda = lambda;
            _partOfTakeIntoAccount = partOfTakeIntoAccount;
        }

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

            var takeIntoAccount = (int)(net.Layers[_ncaLayerIndex].NonBiasNeuronCount*_partOfTakeIntoAccount);

            var algo = new BackpropagationAlgorithm(
                randomizer,
                new CPUAutoencoderNLNCABackpropagationEpocheTrainer(
                    VectorizationSizeEnum.VectorizationMode16,
                    net,
                    config,
                    clProvider,
                    _dodfCalculatorFactory,
                    _ncaLayerIndex,
                    _lambda,
                    takeIntoAccount), 
                mlpContainer,
                net,
                validationDataProvider,
                config);

            return algo;
        }

    }
}
