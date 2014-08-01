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
        private readonly ILayerPropagator _layerPropagator;

        public IMLP MLP
        {
            get;
            private set;
        }

        public CSharpForwardPropagation(
            ILayerPropagator layerPropagator,
            IMLP mlp)
        {
            if (layerPropagator == null)
            {
                throw new ArgumentNullException("layerPropagator");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            _layerPropagator = layerPropagator;

            MLP = mlp;
        }

        public List<ILayerState> ComputeOutput(
            IDataSet dataSet)
        {
            TimeSpan propagationTime;
            var result = ComputeOutput(
                dataSet,
                out propagationTime);

            return result;
        }

        public List<ILayerState> ComputeOutput(
            IDataSet dataSet,
            out TimeSpan propagationTime)
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
                        _layerPropagator.ComputeLayer(
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
                        _layerPropagator.ComputeLayer(
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
    }
//*/
}
