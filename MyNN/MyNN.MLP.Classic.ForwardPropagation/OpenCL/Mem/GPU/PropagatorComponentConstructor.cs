using System;
using System.Collections.Generic;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU
{
    public class PropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private readonly CLProvider _clProvider;

        public PropagatorComponentConstructor(
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

            var ks = new KernelSource();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new LayerPropagator(
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
