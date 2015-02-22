using System;
using System.Collections.Generic;
using MyNN.MLP.DeDyAggregator;
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
            var a = this.CreateAggregators(mlp, c);

            containers = c;
            propagators = p;
            dedyAggregators = a;
        }

        private ICSharpDeDyAggregator[] CreateAggregators(
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

            var layerCount = mlp.Layers.Length;

            var result = new ICSharpDeDyAggregator[layerCount];
            result[0] = null; //для первого слоя нет пропагатора

            for (var layerIndex = layerCount - 1; layerIndex > 0; layerIndex--)
            {
                var previousLayer = mlp.Layers[layerIndex - 1];
                var aggregateLayer = mlp.Layers[layerIndex];

                var p = new CSharpDeDyAggregator(
                    previousLayer.TotalNeuronCount,
                    aggregateLayer.TotalNeuronCount,
                    containers[layerIndex].WeightMem
                    );

                result[layerIndex] = p;
            }

            return
                result;
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
                var currentLayerConfiguration = mlp.Layers[layerIndex].GetConfiguration();

                var mc = new CSharpLayerContainer(
                    currentLayerConfiguration.TotalNeuronCount,
                    currentLayerConfiguration.WeightCount,
                    currentLayerConfiguration.BiasCount
                    );

                result.Add(mc);
            }

            return
                result.ToArray();
        }

    }
}
