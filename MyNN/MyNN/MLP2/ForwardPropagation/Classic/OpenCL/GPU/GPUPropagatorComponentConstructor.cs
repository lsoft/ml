using System;
using System.Collections.Generic;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU
{
    public class GPUPropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private readonly CLProvider _clProvider;

        public GPUPropagatorComponentConstructor(
            CLProvider clProvider
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _clProvider = clProvider;
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

            var c = this.CreateMemsByMLP(mlp);
            var p = this.CreatePropagatorsByMLP(mlp, c);

            containers = c;
            propagators = p;
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

            var ks = new GPUKernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new GPULayerPropagator(
                    _clProvider,
                    ks,
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    mlp.Layers[layerIndex].LayerActivationFunction,
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
                var previousLayerTotalNeuronCount = layerIndex > 0 ? mlp.Layers[layerIndex - 1].Neurons.Length : 0;
                var currentLayerNonBiasNeuronCount = mlp.Layers[layerIndex].NonBiasNeuronCount;
                var currentLayerTotalNeuronCount = mlp.Layers[layerIndex].Neurons.Length;

                var mc = new MemLayerContainer(
                    _clProvider,
                    previousLayerTotalNeuronCount,
                    currentLayerNonBiasNeuronCount,
                    currentLayerTotalNeuronCount
                    );

                result.Add(mc);
            }

            return
                result.ToArray();
        }

    }
}
