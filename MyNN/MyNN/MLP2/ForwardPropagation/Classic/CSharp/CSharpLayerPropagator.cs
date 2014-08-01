using AForge;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp
{
    public class CSharpLayerPropagator : ILayerPropagator
    {
        public CSharpLayerPropagator()
        {
        }

        public float[] ComputeLayer(
            ILayer layer,
            float[] inputVector)
        {
            var lastOutput = new float[layer.Neurons.Length];
            lastOutput[lastOutput.Length - 1] = 1f;

            Parallel.For(0, layer.NonBiasNeuronCount, cc =>
            //for (var cc = 0; cc < layer.NonBiasNeuronCount; cc++)
            {
                var n = layer.Neurons[cc];
                var a = this.Activate(
                    n,
                    inputVector);

                lastOutput[cc] = a;
            }
            ); //Parallel.For

            return
                lastOutput;
        }

        private float Activate(
            INeuron neuron,
            float[] inputVector)
        {
            var sum = this.ComputeNET(
                neuron,
                inputVector);
            
            var lastState = neuron.ActivationFunction.Compute(sum);

            return lastState;
        }

        /// <summary>
        /// Compute NET of the neuron by input vector
        /// </summary>
        /// <param name="neuron">Neuron</param>
        /// <param name="inputVector">Input vector</param>
        /// <returns>Compute NET of neuron</returns>
        private float ComputeNET(
            INeuron neuron,
            float[] inputVector)
        {
            var sum = 0.0f;

            for (var cc = 0; cc < inputVector.Length; ++cc)
            {
                sum += neuron.Weights[cc]*inputVector[cc];
            }

            return
                sum;
        }
    }
}