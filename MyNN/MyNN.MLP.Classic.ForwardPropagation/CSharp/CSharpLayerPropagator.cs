using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Classic.ForwardPropagation.CSharp
{
    public class CSharpLayerPropagator : ILayerPropagator
    {
        private readonly IFullConnectedLayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerContainer;
        private readonly ICSharpLayerContainer _currentLayerContainer;

        public CSharpLayerPropagator(
            ILayer currentLayer,
            ICSharpLayerContainer previousLayerContainer,
            ICSharpLayerContainer currentLayerContainer
            )
        {
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }

            _currentLayer = currentLayer as IFullConnectedLayer;
            _previousLayerContainer = previousLayerContainer;
            _currentLayerContainer = currentLayerContainer;
        }


        public void ComputeLayer()
        {
            Parallel.For(0, _currentLayer.TotalNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < _currentLayer.TotalNeuronCount; neuronIndex++)
            {
                var previousLayerNeuronCountTotal = _previousLayerContainer.Configuration.TotalNeuronCount;//.StateMem.Length;

                var weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                //compute LastNET
                //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

                var acc = new KahanAlgorithm.Accumulator();

                for (var plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
                {
                    var increment = 
                        _currentLayerContainer.WeightMem[weightIndex++]
                        * _previousLayerContainer.StateMem[plnIndex];

                    KahanAlgorithm.AddElement(ref acc, increment);
                }

                var lastNET = acc.Sum + _currentLayerContainer.BiasMem[neuronIndex];

                _currentLayerContainer.NetMem[neuronIndex] = lastNET;

                //compute last state
                var lastState = _currentLayer.LayerActivationFunction.Compute(lastNET);
                _currentLayerContainer.StateMem[neuronIndex] = lastState;
            }
            ); //Parallel.For
        }

        public void WaitForCalculationFinished()
        {
            //nothing to do
        }

        private int ComputeWeightIndex(
           int previousLayerNeuronCount,
           int neuronIndex
            )
        {
            return
                previousLayerNeuronCount * neuronIndex;
        }
    }
}
