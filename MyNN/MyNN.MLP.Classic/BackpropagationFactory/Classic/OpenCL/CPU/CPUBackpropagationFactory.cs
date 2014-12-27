﻿using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU.Backpropagator;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.CPU
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables CPU-OpenCL
    /// </summary>
    public class CPUBackpropagationFactory : IBackpropagationFactory
    {
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly VectorizationSizeEnum _vs;

        public CPUBackpropagationFactory(
            IMLPContainerHelper mlpContainerHelper,
            VectorizationSizeEnum vs
            )
        {
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }

            _mlpContainerHelper = mlpContainerHelper;
            _vs = vs;
        }

        public IBackpropagation CreateBackpropagation(
            IRandomizer randomizer,
            CLProvider clProvider,
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IValidation validationDataProvider,
            ILearningAlgorithmConfig config
            )
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

            var propagatorComponentConstructor = new CPUPropagatorComponentConstructor(
                clProvider,
                _vs
                );

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            propagatorComponentConstructor.CreateComponents(
                mlp,
                out containers,
                out propagators);

            var kernelTextProvider = new MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU.KernelText.KernelTextProvider(mlp, config);

            var desiredValuesContainer = new MemDesiredValuesContainer(clProvider, mlp);

            //создаем бекпропагаторы
            var backpropagators = new IMemLayerBackpropagator[mlp.Layers.Length];
            for (var layerIndex = mlp.Layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                var isLastLayer = layerIndex == mlp.Layers.Length - 1;

                if (isLastLayer)
                {
                    backpropagators[layerIndex] = new CPUOutputLayerBackpropagator(
                        clProvider,
                        mlp,
                        config,
                        containers[layerIndex - 1] as IMemLayerContainer,
                        containers[layerIndex] as IMemLayerContainer,
                        kernelTextProvider,
                        desiredValuesContainer
                        );
                }
                else
                {
                    backpropagators[layerIndex] = new CPUHiddenLayerBackpropagator(
                        clProvider,
                        mlp,
                        config,
                        layerIndex,
                        containers[layerIndex - 1] as IMemLayerContainer,
                        containers[layerIndex] as IMemLayerContainer,
                        containers[layerIndex + 1] as IMemLayerContainer,
                        kernelTextProvider,
                        backpropagators[layerIndex + 1].DeDz
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
                    () => clProvider.QueueFinish(),
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
