using System;
using AForge;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.WeightMask
{
    /// <summary>
    /// Naive implementation of float mask container.
    /// This implementation is inefficient but easy to understand.
    /// </summary>
    public class CSharpWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        private readonly CLProvider _clProvider;
        private readonly IMLPConfiguration _mlpConfiguration;
        private readonly IRandomizer _randomizer;
        private readonly float _p;

        public MemFloat[] MaskMem
        {
            get;
            private set;
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="mlpConfiguration">MLP configuration</param>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public CSharpWeightMaskContainer(
            CLProvider clProvider,
            IMLPConfiguration mlpConfiguration,
            IRandomizer randomizer,
            float p = 0.5f)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlpConfiguration == null)
            {
                throw new ArgumentNullException("mlpConfiguration");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _clProvider = clProvider;
            _mlpConfiguration = mlpConfiguration;

            _randomizer = randomizer;
            _p = p;

            this.CreateInfrastructure();
        }

        private void CreateInfrastructure()
        {
            var layerCount = _mlpConfiguration.Layers.Length;

            MaskMem = new MemFloat[layerCount];

            for (var cc = 1; cc < layerCount; cc++)
            {
                MaskMem[cc] = _clProvider.CreateFloatMem(
                    _mlpConfiguration.Layers[cc].NonBiasNeuronCount * _mlpConfiguration.Layers[cc].Neurons[0].WeightsCount, //without bias neuron at current layer, but include bias neuron at previous layer
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            }
        }

        public void RegenerateMask()
        {
            //надо перезаполнить и записать мем
            
            var layerCount = _mlpConfiguration.Layers.Length;

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
