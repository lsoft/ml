using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.EpocheTrainer.Backpropagator;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Backpropagator;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.DesiredValues;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.BackpropagationFactory.Classic.CSharp
{
    /// <summary>
    /// Factory for classic backpropagation algorithm enables CSharp
    /// </summary>
    public class CSharpBackpropagationFactory : IBackpropagationFactory
    {
        private readonly IMLPContainerHelper _mlpContainerHelper;

        public CSharpBackpropagationFactory(
            IMLPContainerHelper mlpContainerHelper
            )
        {
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }

            _mlpContainerHelper = mlpContainerHelper;
        }

        public IBackpropagation CreateBackpropagation(
            IRandomizer randomizer,
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

            var propagatorComponentConstructor = new CSharpPropagatorComponentConstructor();

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            propagatorComponentConstructor.CreateComponents(
                mlp,
                out containers,
                out propagators);

            var desiredValuesContainer = new CSharpDesiredValuesContainer(mlp);

            //создаем бекпропагаторы
            var backpropagators = new ICSharpLayerBackpropagator[mlp.Layers.Length];
            for (var layerIndex = mlp.Layers.Length - 1; layerIndex > 0; layerIndex--)
            {
                var isLastLayer = layerIndex == mlp.Layers.Length - 1;

                if (isLastLayer)
                {
                    backpropagators[layerIndex] = new CSharpOutputLayerBackpropagator(
                        mlp,
                        config,
                        containers[layerIndex - 1] as ICSharpLayerContainer,
                        containers[layerIndex] as ICSharpLayerContainer,
                        desiredValuesContainer
                        );
                }
                else
                {
                    backpropagators[layerIndex] = new CSharpHiddenLayerBackpropagator(
                        mlp,
                        config,
                        layerIndex,
                        containers[layerIndex - 1] as ICSharpLayerContainer,
                        containers[layerIndex] as ICSharpLayerContainer,
                        containers[layerIndex + 1] as ICSharpLayerContainer,
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
                    () => { },
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
