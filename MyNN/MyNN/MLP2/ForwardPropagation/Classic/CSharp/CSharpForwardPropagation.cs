using System;
using System.Collections.Generic;
using System.Linq;
using AForge;
using MyNN.Data;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.ForwardPropagation.Classic.CSharp
{
    /// <summary>
    /// MLP Forward propagation implemented in parallel C#
    /// </summary>
    public class CSharpForwardPropagation : IForwardPropagation
    {
        public IMLP MLP
        {
            get;
            private set;
        }

        public CSharpForwardPropagation(
            IMLP mlp)
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            MLP = mlp;
        }

        public List<ILayerState> ComputeOutput(IDataSet dataSet)
        {
            TimeSpan propagationTime;
            var result = ComputeOutput(
                dataSet,
                out propagationTime);

            return result;
        }

        public List<ILayerState> ComputeOutput(IDataSet dataSet, out TimeSpan propagationTime)
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            var result = new List<ILayerState>();

            var inputLength = dataSet[0].Input.Length;

            var before = DateTime.Now;

            foreach (var d in dataSet)
            {
                var calcResult = new float[this.MLP.Layers[0].Neurons.Length];
                Array.Copy(d.Input, calcResult, inputLength);
                calcResult[calcResult.Length - 1] = 1f;
                
                for (var cc = 1; cc < this.MLP.Layers.Length; cc++)
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

            var after = DateTime.Now;
            propagationTime = (after - before);

            return result;
        }


        public List<IMLPState> ComputeState(IDataSet dataSet)
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

                var calcResult = new float[this.MLP.Layers[0].Neurons.Length];
                Array.Copy(d.Input, calcResult, inputLength);
                calcResult[calcResult.Length - 1] = 1f;

                for (var cc = 1; cc < this.MLP.Layers.Length; cc++)
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
            ILayer layer,
            float[] inputVector)
        {
            var lastOutput = new float[layer.Neurons.Length];

            //Parallel.For(0, layer.Neurons.Length, cc =>
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
            //); //Parallel.For

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
                sum += neuron.Weights[cc] * inputVector[cc];
            }

            return
                sum;
        }


    }
//*/
}
