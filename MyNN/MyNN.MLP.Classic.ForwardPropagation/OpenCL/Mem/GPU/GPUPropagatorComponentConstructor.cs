using System;
using System.Collections.Generic;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU
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

            var layerCount = mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var p = new LayerPropagator(
                    _clProvider,
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
