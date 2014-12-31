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
        private readonly ILayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerMemContainer;
        private readonly ICSharpLayerContainer _currentLayerMemContainer;

        public CSharpLayerPropagator(
            ILayer currentLayer,
            ICSharpLayerContainer previousLayerMemContainer,
            ICSharpLayerContainer currentLayerMemContainer
            )
        {
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (previousLayerMemContainer == null)
            {
                throw new ArgumentNullException("previousLayerMemContainer");
            }
            if (currentLayerMemContainer == null)
            {
                throw new ArgumentNullException("currentLayerMemContainer");
            }

            _currentLayer = currentLayer;
            _previousLayerMemContainer = previousLayerMemContainer;
            _currentLayerMemContainer = currentLayerMemContainer;
        }


        public void ComputeLayer()
        {
            Parallel.For(0, _currentLayer.TotalNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < _currentLayer.TotalNeuronCount; neuronIndex++)
            {
                var previousLayerNeuronCountTotal = _previousLayerMemContainer.StateMem.Length;

                var weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                //compute LastNET
                //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

                var acc = new KahanAlgorithm.Accumulator();

                for (var plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
                {
                    var increment = 
                        _currentLayerMemContainer.WeightMem[weightIndex++]
                        * _previousLayerMemContainer.StateMem[plnIndex];

                    KahanAlgorithm.AddElement(ref acc, increment);
                }

                var lastNET = acc.Sum + _currentLayerMemContainer.BiasMem[neuronIndex];

                _currentLayerMemContainer.NetMem[neuronIndex] = lastNET;

                //compute last state
                var lastState = _currentLayer.LayerActivationFunction.Compute(lastNET);
                _currentLayerMemContainer.StateMem[neuronIndex] = lastState;
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
