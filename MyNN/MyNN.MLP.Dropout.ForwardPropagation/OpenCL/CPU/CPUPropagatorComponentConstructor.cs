﻿using System;
using System.Collections.Generic;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.DropConnect.WeightMask;
using MyNN.MLP.DropConnect.WeightMask.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.CPU
{
    public class CPUPropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly VectorizationSizeEnum _vse;
        private readonly IOpenCLWeightMaskContainerFactory _maskContainerFactory;

        public CPUPropagatorComponentConstructor(
            IRandomizer randomizer,
            CLProvider clProvider,
            VectorizationSizeEnum vse,
            IOpenCLWeightMaskContainerFactory maskContainerFactory
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

            _randomizer = randomizer;
            _clProvider = clProvider;
            _vse = vse;
            _maskContainerFactory = maskContainerFactory;
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

        private IOpenCLWeightMaskContainer[] CreateMaskContainersByMLP(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var layerCount = mlp.Layers.Length;

            var result = new IOpenCLWeightMaskContainer[layerCount];

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var currentLayerConfiguration = mlp.Layers[layerIndex].GetConfiguration();

                var arraySize = (long) currentLayerConfiguration.NonBiasNeuronCount;

                var maskContainer = _maskContainerFactory.CreateContainer(
                    arraySize
                    );

                result[layerIndex] = maskContainer;
            }

            return
                result;
        }

        private ILayerPropagator[] CreatePropagatorsByMLP(
            IMLP mlp,
            IMemLayerContainer[] containers,
            IOpenCLWeightMaskContainer[] maskContainers
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

            var ks = new CPUKernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new CPULayerPropagator(
                    _randomizer,
                    _clProvider,
                    ks,
                    maskContainers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    mlp.Layers[layerIndex].LayerActivationFunction,
                    _vse,
                    mlp.Layers[layerIndex - 1].Neurons.Length,
                    mlp.Layers[layerIndex].NonBiasNeuronCount
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