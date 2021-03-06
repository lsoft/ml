﻿using System;
using System.Collections.Generic;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.DropConnect.Inferencer;
using MyNN.MLP.DropConnect.Inferencer.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.ForwardPropagation.Inference.OpenCL.GPU
{
    public class PropagatorComponentConstructor : IPropagatorComponentConstructor
    {  
        private readonly CLProvider _clProvider;
        private readonly ILayerInferencerFactory _layerInferencerFactory;

        public PropagatorComponentConstructor(
            CLProvider clProvider,
            ILayerInferencerFactory layerInferencerFactory
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (layerInferencerFactory == null)
            {
                throw new ArgumentNullException("layerInferencerFactory");
            }

            _clProvider = clProvider;
            _layerInferencerFactory = layerInferencerFactory;
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

            var c = this.CreateMemsByMLP(mlp);
            var p = this.CreatePropagatorsByMLP(mlp, c);

            containers = c;
            propagators = p;
            dedyAggregators = null;
        }

        private ILayerPropagator[] CreatePropagatorsByMLP(
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

            var result = new List<ILayerPropagator>();
            result.Add(null); //для первого слоя нет пропагатора

            var ks = new KernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var inferencer = _layerInferencerFactory.CreateLayerInferencer(
                    mlp.Layers[layerIndex - 1],
                    mlp.Layers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex]
                    );

                var p = new LayerPropagator(
                    _clProvider,
                    ks,
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    mlp.Layers[layerIndex].LayerActivationFunction,
                    inferencer
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
