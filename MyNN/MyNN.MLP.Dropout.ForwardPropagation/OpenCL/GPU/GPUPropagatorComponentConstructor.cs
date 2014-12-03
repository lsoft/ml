﻿using System;
using System.Collections.Generic;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.Mask;
using MyNN.Mask.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU
{
    public abstract class GPUPropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private const long Coef = 10L;

        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly float _p;

        protected GPUPropagatorComponentConstructor(
            IRandomizer randomizer,
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
            float p
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
            if (maskContainerFactory == null)
            {
                throw new ArgumentNullException("maskContainerFactory");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _randomizer = randomizer;
            _clProvider = clProvider;
            _maskContainerFactory = maskContainerFactory;
            _p = p;
        }

        public void CreateComponents(
            IMLP mlp,
            out ILayerContainer[] containers,
            out ILayerPropagator[] propagators
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var mc = this.CreateMaskContainersByMLP(mlp);
            var c = this.CreateMemsByMLP(mlp);
            var p = this.CreatePropagatorsByMLP(
                mlp,
                c,
                mc
                );

            containers = c;
            propagators = p;
        }

        private IOpenCLMaskContainer[] CreateMaskContainersByMLP(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var layerCount = mlp.Layers.Length;

            var result = new IOpenCLMaskContainer[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                //это выходной (последний) слой?
                var isOutputLayer = layerIndex == (layerCount - 1);

                var currentLayerConfiguration = mlp.Layers[layerIndex].GetConfiguration();

                var arraySize = (long)currentLayerConfiguration.NonBiasNeuronCount * Coef;

                var maskContainer = _maskContainerFactory.CreateContainer(
                    arraySize,
                    isOutputLayer ? 1f : _p //отключаем маску во внешнем слое
                    );

                result[layerIndex] = maskContainer;
            }

            return
                result;
        }

        private ILayerPropagator[] CreatePropagatorsByMLP(
            IMLP mlp,
            IMemLayerContainer[] containers,
            IOpenCLMaskContainer[] maskContainers
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (containers == null)
            {
                throw new ArgumentNullException("containers");
            }
            if (maskContainers == null)
            {
                throw new ArgumentNullException("maskContainers");
            }

            var result = new List<ILayerPropagator>();
            result.Add(null); //для первого слоя нет пропагатора

            var ks = new GPUKernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                float zeroValue0;
                float oneValue0;
                float zeroValue1;
                float oneValue1;
                GetMaskCoefficient(
                    mlp,
                    layerIndex,
                    out zeroValue0,
                    out oneValue0,
                    out zeroValue1,
                    out oneValue1
                    );

                var p = new GPULayerPropagator(
                    _randomizer,
                    _clProvider,
                    ks,
                    maskContainers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    mlp.Layers[layerIndex].LayerActivationFunction,
                    mlp.Layers[layerIndex - 1].Neurons.Length,
                    mlp.Layers[layerIndex].NonBiasNeuronCount,
                    zeroValue0,
                    oneValue0,
                    zeroValue1,
                    oneValue1
                    );

                result.Add(p);
            }

            return
                result.ToArray();
        }

        protected abstract void GetMaskCoefficient(
            IMLP mlp,
            int layerIndex,
            out float zeroValue0,
            out float oneValue0,
            out float zeroValue1,
            out float oneValue1
            );

        private IMemLayerContainer[] CreateMemsByMLP(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var result = new List<IMemLayerContainer>();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                var currentLayerNonBiasNeuronCount = mlp.Layers[layerIndex].NonBiasNeuronCount;
                var currentLayerTotalNeuronCount = mlp.Layers[layerIndex].Neurons.Length;

                IMemLayerContainer mc;
                if (layerIndex > 0)
                {
                    var previousLayerTotalNeuronCount = mlp.Layers[layerIndex - 1].Neurons.Length;

                    mc = new MemLayerContainer(
                        _clProvider,
                        previousLayerTotalNeuronCount,
                        currentLayerNonBiasNeuronCount,
                        currentLayerTotalNeuronCount
                        );
                }
                else
                {
                    mc = new MemLayerContainer(
                        _clProvider,
                        currentLayerNonBiasNeuronCount,
                        currentLayerTotalNeuronCount
                        );
                }

                result.Add(mc);
            }

            return
                result.ToArray();
        }

    }
}