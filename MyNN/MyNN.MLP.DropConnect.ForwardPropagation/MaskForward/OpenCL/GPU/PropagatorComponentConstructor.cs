﻿using System;
using System.Collections.Generic;
using MyNN.Mask;
using MyNN.Mask.Factory;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU
{
    public class PropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainerFactory _maskContainerFactory;
        private readonly float _p;

        public PropagatorComponentConstructor(
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
            float p
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
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _clProvider = clProvider;
            _maskContainerFactory = maskContainerFactory;
            _p = p;
        }

        public void CreateComponents(
            IMLP mlp,
            out ILayerContainer[] containers,
            out ILayerPropagator[] propagators,
            out IDeDyAggregator[] dedyAggregators
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var mc = this.CreateMaskContainersByMLP(mlp);
            var c = this.CreateMemsByMLP(mlp);
            var p = this.CreatePropagatorsByMLP(mlp, c, mc);
            var a = this.CreateAggregators(mlp, c);

            containers = c;
            propagators = p;
            dedyAggregators = a;
        }

        private IOpenCLDeDyAggregator[] CreateAggregators(
            IMLP mlp,
            IMemLayerContainer[] containers
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

            var layerCount = mlp.Layers.Length;

            var result = new IOpenCLDeDyAggregator[layerCount];
            result[0] = null; //для первого слоя нет аггрегатора

            for (var layerIndex = layerCount - 1; layerIndex > 0; layerIndex--)
            {
                var previousLayer = mlp.Layers[layerIndex - 1];
                var aggregateLayer = mlp.Layers[layerIndex];

                var p = new GPUDeDyAggregator(
                    _clProvider,
                    previousLayer.TotalNeuronCount,
                    aggregateLayer.TotalNeuronCount,
                    containers[layerIndex].WeightMem
                    );

                result[layerIndex] = p;
            }

            return
                result;
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
                var previousLayerConfiguration = mlp.Layers[layerIndex - 1].GetConfiguration();
                var currentLayerConfiguration = mlp.Layers[layerIndex].GetConfiguration();

                var arraySize = (long)currentLayerConfiguration.TotalNeuronCount * (long)previousLayerConfiguration.TotalNeuronCount;

                var maskContainer = _maskContainerFactory.CreateContainer(
                    arraySize,
                    _p
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

            var ks = new KernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new DropConnectLayerPropagator(
                    _clProvider,
                    ks,
                    maskContainers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    mlp.Layers[layerIndex].LayerActivationFunction
                    );

                result.Add(p);
            }

            return
                result.ToArray();
        }

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
                var currentLayerConfiguration = mlp.Layers[layerIndex].GetConfiguration();

                var mc = new MemLayerContainer(
                        _clProvider,
                        currentLayerConfiguration
                        );

                result.Add(mc);
            }

            return
                result.ToArray();
        }

    }
}
