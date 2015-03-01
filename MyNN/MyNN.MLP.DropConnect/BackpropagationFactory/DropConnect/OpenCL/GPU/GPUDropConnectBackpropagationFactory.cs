using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Randomizer;
using MyNN.Mask;
using MyNN.Mask.Factory;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.Backpropagator;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU;
using MyNN.MLP.DropConnect.Inferencer.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.BackpropagationFactory.DropConnect.OpenCL.GPU
{
    /// <summary>
    /// Factory for dropconnect backpropagation algorithm enables GPU-OpenCL
    /// </summary>
    public class GPUDropConnectBackpropagationFactory : IBackpropagationFactory
    {
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly ILayerInferencerFactory _layerInferencerFactory;
        private readonly float _p;
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public GPUDropConnectBackpropagationFactory(
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
            ILayerInferencerFactory layerInferencerFactory,
            float p,
            IMLPContainerHelper mlpContainerHelper
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (maskContainerFactory == null)
            {
                throw new ArgumentNullException("maskContainerFactory");
            }
            if (layerInferencerFactory == null)
            {
                throw new ArgumentNullException("layerInferencerFactory");
            }
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }

            _clProvider = clProvider;
            _maskContainerFactory = maskContainerFactory;
            _layerInferencerFactory = layerInferencerFactory;
            _p = p;
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

            var kernelTextProvider = new MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.KernelText.KernelTextProvider(mlp, config);
            var desiredValuesContainer = new MemDesiredValuesContainer(_clProvider, mlp);
            var backpropagators = new IMemLayerBackpropagator[mlp.Layers.Length];

            IForwardPropagation trainForwardPropagation;
            ILayerContainer[] trainContainers;
            {
                var propagatorComponentConstructor = new MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU.PropagatorComponentConstructor(
                    _clProvider,
                    _maskContainerFactory,
                    _p
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                IDeDyAggregator[] dedyAggregators;
                propagatorComponentConstructor.CreateComponents(
                    mlp,
                    out containers,
                    out propagators,
                    out dedyAggregators
                    );

                trainContainers = containers;

                trainForwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );

                //создаем бекпропагаторы
                for (var layerIndex = mlp.Layers.Length - 1; layerIndex > 0; layerIndex--)
                {
                    var isLastLayer = layerIndex == mlp.Layers.Length - 1;

                    if (isLastLayer)
                    {
                        backpropagators[layerIndex] = new GPUDropConnectOutputLayerBackpropagator(
                            _clProvider,
                            config,
                            containers[layerIndex - 1] as IMemLayerContainer,
                            containers[layerIndex] as IMemLayerContainer,
                            kernelTextProvider,
                            desiredValuesContainer,
                            propagators[layerIndex] as IDropConnectLayerPropagator,
                            dedyAggregators[layerIndex] as IOpenCLDeDyAggregator
                            );
                    }
                    else
                    {
                        backpropagators[layerIndex] = new GPUDropConnectHiddenLayerBackpropagator(
                            _clProvider,
                            config,
                            layerIndex > 1,
                            containers[layerIndex - 1] as IMemLayerContainer,
                            containers[layerIndex] as IMemLayerContainer,
                            containers[layerIndex + 1] as IMemLayerContainer,
                            kernelTextProvider,
                            propagators[layerIndex] as IDropConnectLayerPropagator,
                            dedyAggregators[layerIndex + 1] as IOpenCLDeDyAggregator,
                            dedyAggregators[layerIndex] as IOpenCLDeDyAggregator
                            );
                    }
                }
            }

            IForwardPropagation inferenceForwardPropagation;
            {
                var propagatorComponentConstructor = new MyNN.MLP.DropConnect.ForwardPropagation.Inference.OpenCL.GPU.PropagatorComponentConstructor(
                    _clProvider,
                    _layerInferencerFactory
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                IDeDyAggregator[] dedyAggregators;
                propagatorComponentConstructor.CreateComponents(
                    mlp,
                    out containers,
                    out propagators,
                    out dedyAggregators
                    );

                inferenceForwardPropagation = new MLP.ForwardPropagation.ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );
            }


            var algo = new MLP.Backpropagation.Backpropagation(
                new EpocheTrainer(
                    mlp,
                    config,
                    trainContainers,
                    desiredValuesContainer,
                    backpropagators,
                    () => _clProvider.QueueFinish(),
                    trainForwardPropagation
                    ),
                _mlpContainerHelper,
                artifactContainer,
                mlp,
                validationDataProvider,
                config,
                inferenceForwardPropagation
                );

            return algo;
        }

    }
}
