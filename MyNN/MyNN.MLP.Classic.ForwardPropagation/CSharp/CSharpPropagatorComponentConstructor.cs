using System;
using System.Collections.Generic;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpPropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        public CSharpPropagatorComponentConstructor(
            )
        {
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
            ICSharpLayerContainer[] containers
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

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new CSharpLayerPropagator(
                    mlp.Layers[layerIndex],
                    containers[layerIndex - 1],
                    containers[layerIndex]
                    );

                result.Add(p);
            }

            return
                result.ToArray();
        }

        private ICSharpLayerContainer[] CreateMemsByMLP(
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var result = new List<ICSharpLayerContainer>();

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 0; layerIndex < layerCount; layerIndex++)
            {
                ICSharpLayerContainer mc = null;

                var currentLayerTotalNeuronCount = mlp.Layers[layerIndex].TotalNeuronCount;

                if (layerIndex > 0)
                {
                    var previousLayerTotalNeuronCount = mlp.Layers[layerIndex - 1].TotalNeuronCount;

                    mc = new CSharpLayerContainer(
                        previousLayerTotalNeuronCount,
                        currentLayerTotalNeuronCount
                        );
                }
                else
                {
                    mc = new CSharpLayerContainer(
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
