﻿using System;
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
using MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU;
using MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU.Backpropagator;
using MyNN.MLP.Dropout.ForwardPropagation.OpenCL;
using MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
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
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly float _p;
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public GPUDropoutBackpropagationFactory(
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
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
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }

            _clProvider = clProvider;
            _maskContainerFactory = maskContainerFactory;
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

            var propagatorComponentConstructor = new GPUMaskForwardPropagatorComponentConstructor(
                randomizer,
                _clProvider,
                _maskContainerFactory,
                _p
                );

            var kernelTextProvider = new MyNN.MLP.Dropout.Backpropagation.EpocheTrainer.Dropout.OpenCL.GPU.KernelText.KernelTextProvider(mlp, config);
            var desiredValuesContainer = new MemDesiredValuesContainer(_clProvider, mlp);
            var backpropagators = new IMemLayerBackpropagator[mlp.Layers.Length];

            IForwardPropagation trainForwardPropagation;
            ILayerContainer[] trainContainers;
            {
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
                        backpropagators[layerIndex] = new GPUDropoutOutputLayerBackpropagator(
                            _clProvider,
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
                        backpropagators[layerIndex] = new GPUDropoutHiddenLayerBackpropagator(
                            _clProvider,
                            config,
                            layerIndex > 1,
                            containers[layerIndex - 1] as IMemLayerContainer,
                            containers[layerIndex] as IMemLayerContainer,
                            containers[layerIndex + 1] as IMemLayerContainer,
                            kernelTextProvider,
                            propagators[layerIndex] as IDropoutLayerPropagator,
                            dedyAggregators[layerIndex + 1] as IOpenCLDeDyAggregator,
                            dedyAggregators[layerIndex] as IOpenCLDeDyAggregator
                            );
                    }
                }
            }

            IForwardPropagation inferenceForwardPropagation;
            {
                var cc = new GPUInferencePropagatorComponentConstructor(
                    randomizer,
                    _clProvider,
                    _maskContainerFactory,
                    _p
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
