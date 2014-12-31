using System;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DropConnect.Inferencer.CSharp
{
    /// <summary>
    /// Default implementation of layer inferencer on CSharp.
    /// It is very inefficient, but easy to understand.
    /// </summary>
    public class NaiveLayerInferencer : ILayerInferencer
    {

        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        private readonly MemFloat _weightMem;
        private readonly MemFloat _previousLayerStateMem;
        private readonly MemFloat _currentLayerStateMem;
        private readonly float _p;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="previousLayer">Previous layer of dropconnect MLP (the algorithm needs to know neuron count of previous layer)</param>
        /// <param name="currentLayer">Current layer of dropconnect MLP (the algorithm needs to know neuron count and activation function of current layer)</param>
        /// <param name="weightMem">Weights of current MLP layer</param>
        /// <param name="previousLayerStateMem">State of previous layer neurons</param>
        /// <param name="currentLayerStateMem">State of current layer neurons</param>
        /// <param name="p">Probability for each bit to be ONE (TRUE) (with p = 1 it completely disables mask and convert the model to classic backprop)</param>
        public NaiveLayerInferencer(
            IRandomizer randomizer,
            CLProvider clProvider,
            int sampleCount,
            ILayer previousLayer,
            ILayer currentLayer,
            MemFloat weightMem,
            MemFloat previousLayerStateMem,
            MemFloat currentLayerStateMem,
            float p
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (previousLayer == null)
            {
                throw new ArgumentNullException("previousLayer");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (weightMem == null)
            {
                throw new ArgumentNullException("weightMem");
            }
            if (previousLayerStateMem == null)
            {
                throw new ArgumentNullException("previousLayerStateMem");
            }
            if (currentLayerStateMem == null)
            {
                throw new ArgumentNullException("currentLayerStateMem");
            }

            _randomizer = randomizer;
            _clProvider = clProvider;
            _sampleCount = sampleCount;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _weightMem = weightMem;
            _previousLayerStateMem = previousLayerStateMem;
            _currentLayerStateMem = currentLayerStateMem;
            _p = p;

        }

        public void InferenceLayer()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //read all data from OpenCL
            this._previousLayerStateMem.Read(BlockModeEnum.NonBlocking);
            this._weightMem.Read(BlockModeEnum.NonBlocking);
            this._currentLayerStateMem.Read(BlockModeEnum.NonBlocking);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //calculate inference

            var previousLayerNeuronCountTotal = this._previousLayer.TotalNeuronCount;

            Parallel.For(0, _currentLayer.TotalNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < _currentLayer.NonBiasNeuronCount; neuronIndex++)
            {
                var normal = new Normal(0, 1);
                normal.RandomSource = new Random(_randomizer.Next(100000));

                //суммируем веса * состояние нейронов пред. слоя и высчитываем медиану и сигма-квадрат для гауссианы
                var weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);




                //float wv_median  = 0;
                //float wv_sigmasq = 0;
                //for (var plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
                //{
                //    var wv = this._weightMem.Array[weightIndex++] * this._previousLayerStateMem.Array[plnIndex];

                //    wv_median += wv;
                //    wv_sigmasq += wv * wv;
                //}



                var accMedian = new KahanAlgorithm.Accumulator();
                var accSigmaSq = new KahanAlgorithm.Accumulator();

                for (var plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
                {
                    var wv = this._weightMem.Array[weightIndex++] * this._previousLayerStateMem.Array[plnIndex];

                    KahanAlgorithm.AddElement(ref accMedian, wv);
                    KahanAlgorithm.AddElement(ref accSigmaSq, wv * wv);
                }

                var wv_median = accMedian.Sum;
                var wv_sigmasq = accSigmaSq.Sum;




                wv_median *= this._p;
                wv_sigmasq *= this._p * (1 - this._p);

                var wv_sigma = (float)Math.Sqrt(wv_sigmasq);


                


                //var lastStateSummator  = 0f;
                //for(var sampleIndex = 0; sampleIndex < this._sampleCount; sampleIndex++)
                //{
                //    //делаем гауссиану с медианой wv_median и сигмой wv_sigma из гауссианы (0;1), пришедшей из C#
                //    var ogrnd = (float)normal.Sample();
                //    var grnd = ogrnd * wv_sigma + wv_median;

                //    //compute last state
                //    var lastState = _currentLayer.LayerActivationFunction.Compute(grnd);

                //    lastStateSummator += lastState;
                //}


                var lastStateSummator = KahanAlgorithm.Sum(
                    this._sampleCount,
                    sampleIndex => 
                {
                    //делаем гауссиану с медианой wv_median и сигмой wv_sigma из гауссианы (0;1), пришедшей из C#
                    var ogrnd = (float)normal.Sample();
                    var grnd = ogrnd * wv_sigma + wv_median;

                    //compute last state
                    var lastState = _currentLayer.LayerActivationFunction.Compute(grnd);

                    return
                        lastState;
                }
                ); //Kahan.Sum




                //усредняем
                var result = lastStateSummator / this._sampleCount;

                //записываем обратно в хранилище
                this._currentLayerStateMem.Array[neuronIndex] = result;
            }
            ); //Parallel.For

            //write all data back to OpenCL

            this._previousLayerStateMem.Write(BlockModeEnum.NonBlocking);
            this._weightMem.Write(BlockModeEnum.NonBlocking);
            this._currentLayerStateMem.Write(BlockModeEnum.NonBlocking);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        private int ComputeWeightIndex(
            int previousLayerNeuronCount,
            int neuronIndex)
        {
            return
                previousLayerNeuronCount*neuronIndex;
        }

    }
}
