using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.ForwardPropagation
{
    public class CCharpForwardPropagation : IForwardPropagation
    {
        public MLP MLP
        {
            get;
            private set;
        }

        public CCharpForwardPropagation(
            MLP mlp)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            MLP = mlp;

            throw new Exception("юзатель! помни! этот форвардер не проверен ни разу! написан вслепую! прежде чем юзать - зотести!");
        }

        public List<ILayerState> ComputeOutput(DataSet dataSet)
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            var result = new List<ILayerState>();

            var inputLength = dataSet[0].Input.Length;

            foreach (var d in dataSet)
            {
                var calcResult = new float[inputLength];
                Array.Copy(d.Input, calcResult, inputLength);
                
                for (var cc = 0; cc < this.MLP.Layers.Length; cc++)
                {
                    var tmp =
                        this.ComputeLayer(
                            MLP.Layers[cc],
                            calcResult);

                    calcResult = new float[tmp.Length];
                    Array.Copy(tmp, calcResult, tmp.Length);
                }

                var ls = new LayerState(
                    calcResult,
                    this.MLP.Layers.Last().NonBiasNeuronCount);
                result.Add(ls);
            }

            return result;
        }

        public List<IMLPState> ComputeState(DataSet dataSet)
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            var result = new List<IMLPState>();

            var inputLength = dataSet[0].Input.Length;

            foreach (var d in dataSet)
            {
                var listls = new List<ILayerState>();

                var calcResult = new float[inputLength];
                Array.Copy(d.Input, calcResult, inputLength);

                for (var cc = 0; cc < this.MLP.Layers.Length; cc++)
                {
                    var tmp =
                        this.ComputeLayer(
                            MLP.Layers[cc],
                            calcResult);

                    calcResult = new float[tmp.Length];
                    Array.Copy(tmp, calcResult, tmp.Length);

                    var ls = new LayerState(
                        calcResult,
                         MLP.Layers[cc].NonBiasNeuronCount);

                    listls.Add(ls);
                }

                result.Add(
                    new MLPState(listls.ToArray()));
            }

            return result;
        }

        private float[] ComputeLayer(
            MLPLayer layer,
            float[] inputVector)
        {
            var lastOutput = new float[layer.Neurons.Length];

            for (var cc = 0; cc < layer.Neurons.Length; cc++)
            {
                var a = 1f;

                var n = layer.Neurons[cc];
                //if (!n.IsBiasNeuron)
                //{
                    a = this.Activate(
                        n,
                        inputVector);
                //}

                lastOutput[cc] = a;
            }

            return
                lastOutput;
        }

        private float Activate(
            TrainableMLPNeuron neuron,
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
            TrainableMLPNeuron neuron,
            float[] inputVector)
        {
            var sum = 0.0f;

            for (var cc = 0; cc < inputVector.Length; ++cc)
            {
                sum += neuron.Weights[cc] * inputVector[cc];
            }

            return
                sum;
        }


    }
}
