using System;
using AForge;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnect.WeightMask
{
    public class CPUWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        private readonly CLProvider _clProvider;
        private readonly MLP _mlp;
        private readonly IRandomizer _randomizer;
        private readonly float _p;

        public MemFloat[] MaskMem
        {
            get;
            private set;
        }

        public CPUWeightMaskContainer(
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

            MaskMem = new MemFloat[layerCount];

            for (var cc = 1; cc < layerCount; cc++)
            {
                MaskMem[cc] = _clProvider.CreateFloatMem(
                    _mlp.Layers[cc].NonBiasNeuronCount * _mlp.Layers[cc].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            }
        }

        public void RegenerateMask()
        {
            //надо перезаполнить и записать мем
            
            var layerCount = _mlp.Layers.Length;

            Parallel.For(1, layerCount, cc =>
            //for (var cc = 1; cc < layerCount; cc++)
            {
                for (var i = 0; i < this.MaskMem[cc].Array.Length; i++)
                {
                    this.MaskMem[cc].Array[i] = _randomizer.Next() < _p ? 1f : 0f;
                }

                MaskMem[cc].Write(BlockModeEnum.NonBlocking);
            }
            ); //Parallel.For

        }
    }
}
