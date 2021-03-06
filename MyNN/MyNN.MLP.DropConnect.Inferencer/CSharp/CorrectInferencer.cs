﻿using System;
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
    /// Correct inferencer without gaussian approximation
    /// </summary>
    public class CorrectInferencer : ILayerInferencer
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
        public CorrectInferencer(
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
                var b = new Bernoulli(this._p);
                b.RandomSource = new Random(_randomizer.Next(100000));

                var weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                var lastStateSummator = KahanAlgorithm.Sum(
                    _sampleCount,
                    cc =>
                    {
                        var sum = KahanAlgorithm.Sum(
                            previousLayerNeuronCountTotal,
                            (plnIndex) =>
                            {
                                var wv =
                                    this._weightMem.Array[weightIndex + plnIndex]
                                    * this._previousLayerStateMem.Array[plnIndex]
                                    * b.Sample();

                                return
                                    wv;
                            });

                        //compute last state
                        var lastState = _currentLayer.LayerActivationFunction.Compute(sum);

                        return 
                            lastState;
                });

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
                previousLayerNeuronCount * neuronIndex;
        }

    }
}
