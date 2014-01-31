using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using AForge;
using AForge.Math.Metrics;
using MathNet.Numerics.Distributions;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnect.WeightMask
{
    public class BGWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        private const int MaskCount = 50;

        private readonly CLProvider _clProvider;
        private readonly MLP _mlp;
        private readonly IRandomizer _randomizer;
        private readonly float _p;

        private List<MemFloat[]> _maskMem;
        private int _currentIndex;

        public MemFloat[] MaskMem
        {
            get
            {
                return
                    _maskMem[_currentIndex];
            }
        }

        public BGWeightMaskContainer(
            CLProvider clProvider,
            MLP mlp,
            IRandomizer randomizer,
            float p = 0.5f)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _clProvider = clProvider;
            _mlp = mlp;

            _randomizer = randomizer;
            _p = p;

            this.CreateInfrastructure();
        }

        private void CreateInfrastructure()
        {
            var layerCount = _mlp.Layers.Length;

            _maskMem = new List<MemFloat[]>();

            for (var mc = 0; mc < MaskCount; mc++)
            {
                var masks = new MemFloat[layerCount];

                for (var cc = 1; cc < layerCount; cc++)
                {
                    masks[cc] = _clProvider.CreateFloatMem(
                        _mlp.Layers[cc].NonBiasNeuronCount * _mlp.Layers[cc].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                        MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                }

                _maskMem.Add(masks);
            }

            _currentIndex = -1;
        }

        public void RegenerateMask()
        {
            var newIndex = _currentIndex + 1;

            if(newIndex >= MaskCount)
            {
                InternalRegenerate();
                
                newIndex = 0;
            }

            _currentIndex = newIndex;
        }

        private void InternalRegenerate()
        {
            var layerCount = _mlp.Layers.Length;

            Parallel.For(0, MaskCount, ni =>
            //for (var ni = 0; ni < MaskCount; ni++)
            {
                var brnd = new Bernoulli(_p)
                {
                    RandomSource = new Random(_randomizer.Next(1000000))
                };

                for (var li = 1; li < layerCount; li++)
                {
                    var mem = this._maskMem[ni][li];

                    for (var i = 0; i < mem.Array.Length; i++)
                    {
                        mem.Array[i] = brnd.Sample();
                    }

                   mem.Write(BlockModeEnum.NonBlocking);
                }
            }
            ); //Parallel.For
        }
    }
}
