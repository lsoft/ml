using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.NLNCA.BackpropagationFactory.OpenCL.CPU
{
    /// <summary>
    /// Factory for NLNCA-backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class CPUNLNCAClassificationBackpropagationFactory : IBackpropagationFactory
    {
        private readonly CLProvider _clProvider;
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly Func<List<IDataItem>, IDodfCalculator> _dodfCalculatorFactory;
        private readonly VectorizationSizeEnum _vse;

        public CPUNLNCAClassificationBackpropagationFactory(
            CLProvider clProvider,
            IMLPContainerHelper mlpContainerHelper,
            Func<List<IDataItem>, IDodfCalculator> dodfCalculatorFactory,
            VectorizationSizeEnum vse
            )
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
            _vse = vse;
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

            var cc = new MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU.CPUPropagatorComponentConstructor(
                _clProvider,
                _vse
                );

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            IDeDyAggregator[] dedyAggregators;
            cc.CreateComponents(
                mlp,
                out containers,
                out propagators,
                out dedyAggregators
                );

            var forwardPropagation = new ForwardPropagation.ForwardPropagation(
                containers,
                propagators,
                mlp
                );

            var algo = new MLP.Backpropagation.Backpropagation(
                new CPUNLNCAEpocheTrainer(
                    mlp,
                    config,
                    _clProvider,
                    _dodfCalculatorFactory,
                    forwardPropagation,
                    containers.ConvertAll(j => j as IMemLayerContainer)
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
