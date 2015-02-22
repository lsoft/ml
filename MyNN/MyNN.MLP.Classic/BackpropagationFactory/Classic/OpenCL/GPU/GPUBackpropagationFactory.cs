using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU.Backpropagator;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.GPU
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables GPU-OpenCL
    /// </summary>
    public class GPUBackpropagationFactory : IBackpropagationFactory
    {
        private readonly CLProvider _clProvider;
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public GPUBackpropagationFactory(
            CLProvider clProvider,
            IMLPContainerHelper mlpContainerHelper
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

            _clProvider = clProvider;
            _mlpContainerHelper = mlpContainerHelper;
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

            var propagatorComponentConstructor = new GPUPropagatorComponentConstructor(_clProvider);

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            IDeDyAggregator[] dedyAggregators;
            propagatorComponentConstructor.CreateComponents(
                mlp,
                out containers,
                out propagators,
                out dedyAggregators
                );

            var kernelTextProvider = new MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU.KernelText.KernelTextProvider(mlp, config);

            var desiredValuesContainer = new MemDesiredValuesContainer(_clProvider, mlp);

            //создаем бекпропагаторы
            var backpropagators = new IMemLayerBackpropagator[mlp.Layers.Length];
            for (var layerIndex = mlp.Layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                var isLastLayer = layerIndex == mlp.Layers.Length - 1;

                if (isLastLayer)
                {
                    backpropagators[layerIndex] = new GPUOutputLayerBackpropagator(
                        _clProvider,
                        mlp,
                        config,
                        containers[layerIndex - 1] as IMemLayerContainer,
                        containers[layerIndex] as IMemLayerContainer,
                        kernelTextProvider,
                        desiredValuesContainer,
                        dedyAggregators[layerIndex] as IOpenCLDeDyAggregator
                        );
                }
                else
                {
                    backpropagators[layerIndex] = new GPUHiddenLayerBackpropagator(
                        _clProvider,
                        mlp,
                        config,
                        layerIndex,
                        containers[layerIndex - 1] as IMemLayerContainer,
                        containers[layerIndex] as IMemLayerContainer,
                        containers[layerIndex + 1] as IMemLayerContainer,
                        kernelTextProvider,
                        dedyAggregators[layerIndex + 1] as IOpenCLDeDyAggregator,
                        dedyAggregators[layerIndex] as IOpenCLDeDyAggregator
                        );
                }
            }

            var forwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                containers,
                propagators,
                mlp
                );

            var algo = new MLP.Backpropagation.Backpropagation(
                new EpocheTrainer(
                    mlp,
                    config,
                    containers,
                    desiredValuesContainer,
                    backpropagators,
                    () => _clProvider.QueueFinish(),
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
