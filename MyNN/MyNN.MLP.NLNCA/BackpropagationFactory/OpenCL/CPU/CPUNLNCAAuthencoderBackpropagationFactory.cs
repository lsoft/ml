using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.AutoencoderMLP.OpenCL.CPU;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.NLNCA.BackpropagationFactory.OpenCL.CPU
{
    /// <summary>
    /// Factory for NLNCA-backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class CPUNLNCAAuthencoderBackpropagationFactory : IBackpropagationFactory
    {
        private readonly CLProvider _clProvider;
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly Func<List<IDataItem>, IDodfCalculator> _dodfCalculatorFactory;
        private readonly int _ncaLayerIndex;
        private readonly float _lambda;
        private readonly float _partOfTakeIntoAccount;

        public CPUNLNCAAuthencoderBackpropagationFactory(
            CLProvider clProvider,
            IMLPContainerHelper mlpContainerHelper,
            Func<List<IDataItem>, IDodfCalculator> dodfCalculatorFactory,
            int ncaLayerIndex,
            float lambda,
            float partOfTakeIntoAccount)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            if (dodfCalculatorFactory == null)
            {
                throw new ArgumentNullException("dodfCalculatorFactory");
            }

            _clProvider = clProvider;
            _mlpContainerHelper = mlpContainerHelper;
            _dodfCalculatorFactory = dodfCalculatorFactory;
            _ncaLayerIndex = ncaLayerIndex;
            _lambda = lambda;
            _partOfTakeIntoAccount = partOfTakeIntoAccount;
        }

        public IBackpropagation CreateBackpropagation(
            IRandomizer randomizer,
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (_clProvider == null)
            {
                throw new ArgumentNullException("_clProvider");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (validationDataProvider == null)
            {
                throw new ArgumentNullException("validationDataProvider");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            var takeIntoAccount = (int)(mlp.Layers[_ncaLayerIndex].NonBiasNeuronCount*_partOfTakeIntoAccount);

            var cc = new CPUPropagatorComponentConstructor(
                _clProvider,
                VectorizationSizeEnum.VectorizationMode16
                );

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            cc.CreateComponents(
                mlp,
                out containers,
                out propagators);

            var forwardPropagation = new ForwardPropagation.ForwardPropagation(
                containers,
                propagators,
                mlp
                );

            var algo = new MLP.Backpropagation.Backpropagation(
                new CPUAutoencoderNLNCAEpocheTrainer(
                    mlp,
                    config,
                    _clProvider,
                    _dodfCalculatorFactory,
                    _ncaLayerIndex,
                    _lambda,
                    takeIntoAccount,
                    forwardPropagation
                    ), 
                _mlpContainerHelper,
                artifactContainer,
                mlp,
                validationDataProvider,
                config,
                forwardPropagation
                );

            return algo;
        }

    }
}
