using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp.Two
{
    public class CSharpCPULayerPropagator : ICPULayerPropagator
    {
        private readonly ILayer _currentLayer;
        private readonly ICSharpLayerContainer _previousLayerMemContainer;
        private readonly ICSharpLayerContainer _currentLayerMemContainer;

        public CSharpCPULayerPropagator(
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
            Parallel.For(0, _currentLayer.NonBiasNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < currentLayer.NonBiasNeuronCount; neuronIndex++)
            {
                var n = _currentLayer.Neurons[neuronIndex];
                this.ActivateNeuron(neuronIndex, n);
            }
            ); //Parallel.For
        }

        public void WaitForCalculationFinished()
        {
            //nothing to do
        }

        #region private methods

        private void ActivateNeuron(
            int neuronIndex,
            INeuron neuron)
        {
            var lastNet = this.ComputeNet(neuron);
            var lastState = neuron.ActivationFunction.Compute(lastNet);

            _currentLayerMemContainer.NetMem[neuronIndex] = lastNet;
            _currentLayerMemContainer.StateMem[neuronIndex] = lastState;
        }

        /// <summary>
        /// Compute NET of the neuron by input vector
        /// </summary>
        /// <param name="neuron">Neuron</param>
        /// <returns>Compute NET of neuron</returns>
        private float ComputeNet(INeuron neuron)
        {
            var inputVector = _previousLayerMemContainer.StateMem;

            var sum = 0.0f;

            for (var cc = 0; cc < inputVector.Length; ++cc)
            {
                sum += neuron.Weights[cc]*inputVector[cc];
            }

            return 
                sum;
        }

        #endregion
    }
}
