using System;
using System.Collections.Generic;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU.Two
{
    public class GPUPropagatorComponentConstructor
    {
        private readonly CLProvider _clProvider;
        private readonly IMLP _mlp;

        public GPUPropagatorComponentConstructor(
            CLProvider clProvider,
            IMLP mlp
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            _clProvider = clProvider;
            _mlp = mlp;
        }

        public void CreateComponents(
            out IMemLayerContainer[] containers,
            out ILayerPropagator[] propagators
            )
        {
            var c = this.CreateMemsByMLP();
            var p = this.CreatePropagatorsByMLP(c);

            containers = c;
            propagators = p;
        }

        private ILayerPropagator[] CreatePropagatorsByMLP(
            IMemLayerContainer[] containers
            )
        {
            if (containers == null)
            {
                throw new ArgumentNullException("containers");
            }

            var result = new List<ILayerPropagator>();
            result.Add(null); //для первого слоя нет пропагатора

            var ks = new GPUKernelSource();

            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new GPULayerPropagator(
                    _clProvider,
                    ks,
                    containers[layerIndex - 1],
                    containers[layerIndex],
                    _mlp.Layers[layerIndex].LayerActivationFunction,
                    _mlp.Layers[layerIndex - 1].Neurons.Length,
                    _mlp.Layers[layerIndex].NonBiasNeuronCount
                    );

                result.Add(p);
            }

            return
                result.ToArray();
        }

        private IMemLayerContainer[] CreateMemsByMLP(
            )
        {
            var result = new List<IMemLayerContainer>();

            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                var previousLayerTotalNeuronCount = layerIndex > 0 ? _mlp.Layers[layerIndex - 1].Neurons.Length : 0;
                var currentLayerNonBiasNeuronCount = _mlp.Layers[layerIndex].NonBiasNeuronCount;
                var currentLayerTotalNeuronCount = _mlp.Layers[layerIndex].Neurons.Length;

                var mc = new CPUMemLayerContainer(
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
