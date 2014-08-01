using System;
using System.Collections.Generic;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp
{
    public class CSharpPropagatorComponentConstructor
    {
        private readonly IMLP _mlp;

        public CSharpPropagatorComponentConstructor(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            _mlp = mlp;
        }

        public void CreateComponents(
            out ICSharpLayerContainer[] containers,
            out ILayerPropagator[] propagators
            )
        {
            var c = this.CreateMemsByMLP();
            var p = this.CreatePropagatorsByMLP(c);

            containers = c;
            propagators = p;
        }

        private ILayerPropagator[] CreatePropagatorsByMLP(
            ICSharpLayerContainer[] containers
            )
        {
            if (containers == null)
            {
                throw new ArgumentNullException("containers");
            }

            var result = new List<ILayerPropagator>();
            result.Add(null); //для первого слоя нет пропагатора

            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new CSharpLayerPropagator(
                    _mlp.Layers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex]
                    );

                result.Add(p);
            }

            return
                result.ToArray();
        }

        private ICSharpLayerContainer[] CreateMemsByMLP(
            )
        {
            var result = new List<ICSharpLayerContainer>();

            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                var previousLayerTotalNeuronCount = layerIndex > 0 ? _mlp.Layers[layerIndex - 1].Neurons.Length : 0;
                var currentLayerNonBiasNeuronCount = _mlp.Layers[layerIndex].NonBiasNeuronCount;
                var currentLayerTotalNeuronCount = _mlp.Layers[layerIndex].Neurons.Length;

                var mc = new CSharpLayerContainer(
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
