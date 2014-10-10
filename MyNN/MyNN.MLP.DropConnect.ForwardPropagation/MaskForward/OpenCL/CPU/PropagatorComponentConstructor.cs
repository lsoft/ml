using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.CPU.LayerPropagator;
using MyNN.MLP.DropConnect.WeightMask;
using MyNN.MLP.DropConnect.WeightMask.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.CPU
{
    public class PropagatorComponentConstructor : IPropagatorComponentConstructor
    {
        private readonly CLProvider _clProvider;
        private readonly IOpenCLWeightMaskContainerFactory _maskContainerFactory;

        public PropagatorComponentConstructor(
            CLProvider clProvider,
            IOpenCLWeightMaskContainerFactory maskContainerFactory
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

            _clProvider = clProvider;
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
            var p = this.CreatePropagatorsByMLP(mlp, c, mc);

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
                var maskContainer = _maskContainerFactory.CreateContainer(
                    mlp.Layers[layerIndex - 1].GetConfiguration(),
                    mlp.Layers[layerIndex].GetConfiguration()
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
