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
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly ILayerInferencerFactory _layerInferencerFactory;
        private readonly float _p;
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public GPUDropConnectBackpropagationFactory(
            IOpenCLMaskContainerFactory maskContainerFactory,
            ILayerInferencerFactory layerInferencerFactory,
            float p,
            IMLPContainerHelper mlpContainerHelper
            )
        {
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

            _maskContainerFactory = maskContainerFactory;
            _layerInferencerFactory = layerInferencerFactory;
            _p = p;
            _mlpContainerHelper = mlpContainerHelper;
        }

        public IBackpropagation CreateBackpropagation(
            IRandomizer randomizer,
            CLProvider clProvider,
            IArtifactContainer artifactContainer,
            IMLP mlp,
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

            var propagatorComponentConstructor = new MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU.PropagatorComponentConstructor(
                clProvider,
                _maskContainerFactory,
                _p
                );

            var kernelTextProvider = new MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU.KernelText.KernelTextProvider(mlp, config);
            var desiredValuesContainer = new MemDesiredValuesContainer(clProvider, mlp);
            var backpropagators = new IMemLayerBackpropagator[mlp.Layers.Length];

            IForwardPropagation trainForwardPropagation;
            ILayerContainer[] trainContainers;
            {
                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                propagatorComponentConstructor.CreateComponents(
                    mlp,
                    out containers,
                    out propagators);

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
                            clProvider,
                            mlp,
                            config,
                            containers[layerIndex - 1] as IMemLayerContainer,
                            containers[layerIndex] as IMemLayerContainer,
                            kernelTextProvider,
                            desiredValuesContainer,
                            propagators[layerIndex] as IDropConnectLayerPropagator
                            );
                    }
                    else
                    {
                        backpropagators[layerIndex] = new GPUDropConnectHiddenLayerBackpropagator(
                            clProvider,
                            mlp,
                            config,
                            layerIndex,
                            containers[layerIndex - 1] as IMemLayerContainer,
                            containers[layerIndex] as IMemLayerContainer,
                            containers[layerIndex + 1] as IMemLayerContainer,
                            kernelTextProvider,
                            backpropagators[layerIndex + 1].DeDz,
                            propagators[layerIndex] as IDropConnectLayerPropagator
                            );
                    }
                }
            }

            IForwardPropagation inferenceForwardPropagation;
            {
                var cc = new MyNN.MLP.DropConnect.ForwardPropagation.Inference.OpenCL.GPU.PropagatorComponentConstructor(
                    clProvider,
                    _layerInferencerFactory
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                cc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators);

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
                    () => clProvider.QueueFinish(),
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
