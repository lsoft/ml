﻿using System;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL
{
    public class MemLayerContainer : IMemLayerContainer
    {
        private readonly CLProvider _clProvider;
        private readonly int _previousLayerTotalNeuronCount;
        private readonly int _currentLayerNonBiasNeuronCount;
        private readonly int _currentLayerTotalNeuronCount;

        public MemFloat WeightMem
        {
            get;
            private set;
        }

        public MemFloat NetMem
        {
            get;
            private set;
        }

        public MemFloat StateMem
        {
            get;
            private set;
        }

        public MemLayerContainer(
            CLProvider clProvider,
            int currentLayerNonBiasNeuronCount,
            int currentLayerTotalNeuronCount
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _clProvider = clProvider;
            _previousLayerTotalNeuronCount = 0;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;

            //нейроны
            var netMem = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            netMem.Write(BlockModeEnum.Blocking);

            NetMem = netMem;

            var stateMem = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            stateMem.Write(BlockModeEnum.Blocking);

            StateMem = stateMem;
        }

        public MemLayerContainer(
            CLProvider clProvider,
            int previousLayerTotalNeuronCount,
            int currentLayerNonBiasNeuronCount,
            int currentLayerTotalNeuronCount
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (previousLayerTotalNeuronCount == 0)
            {
                throw new ArgumentException("previousLayerTotalNeuronCount == 0");
            }

            _clProvider = clProvider;
            _previousLayerTotalNeuronCount = previousLayerTotalNeuronCount;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;

            //нейроны
            var netMem = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            netMem.Write(BlockModeEnum.Blocking);

            NetMem = netMem;

            var stateMem = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            stateMem.Write(BlockModeEnum.Blocking);

            StateMem = stateMem;

            //веса
            var weightMem = clProvider.CreateFloatMem(
                currentLayerNonBiasNeuronCount*previousLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            weightMem.Write(BlockModeEnum.Blocking);

            WeightMem = weightMem;
        }

        public void ClearAndPushHiddenLayers()
        {
            ClearHiddenLayers();

            PushHiddenLayers();
        }

        public void ClearHiddenLayers()
        {
            var nml = this.NetMem.Array.Length;
            Array.Clear(this.NetMem.Array, 0, nml);
            this.NetMem.Array[nml - 1] = 1f;

            var sml = this.StateMem.Array.Length;
            Array.Clear(this.StateMem.Array, 0, sml);
            this.StateMem.Array[sml - 1] = 1f;
        }


        public void PushHiddenLayers()
        {
            this.NetMem.Write(BlockModeEnum.NonBlocking);

            this.StateMem.Write(BlockModeEnum.NonBlocking);
        }

        public void PushInput(float[] data)
        {
            //записываем значения из сети в объекты OpenCL
            for (var neuronIndex = 0; neuronIndex < _currentLayerTotalNeuronCount; neuronIndex++)
            {
                var isBiasNeuron = neuronIndex == _currentLayerNonBiasNeuronCount;

                this.NetMem.Array[neuronIndex] = 0; //LastNET
                this.StateMem.Array[neuronIndex] =
                    isBiasNeuron
                        ? 1.0f
                        : data[neuronIndex];
            }

            this.NetMem.Write(BlockModeEnum.NonBlocking);
            this.StateMem.Write(BlockModeEnum.NonBlocking);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        public void PushWeights(ILayer layer)
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            var weightShift = 0;

            var weightMem = this.WeightMem;
            for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                Array.Copy(
                    neuron.Weights,
                    0,
                    weightMem.Array,
                    weightShift,
                    neuron.Weights.Length);

                weightShift += neuron.Weights.Length;
            }

            weightMem.Write(BlockModeEnum.NonBlocking);
            
        }

        public void PopHiddenState()
        {
            //читаем его из opencl
            this.NetMem.Read(BlockModeEnum.Blocking);
            this.StateMem.Read(BlockModeEnum.Blocking);
        }

        public void PopLastLayerState()
        {
            //извлекаем из Opencl последний слой
            this.NetMem.Read(BlockModeEnum.Blocking);
            this.StateMem.Read(BlockModeEnum.Blocking);
        }

        public ILayerState GetLayerState()
        {
            var ls = new LayerState(
                this.StateMem.Array,
                _currentLayerNonBiasNeuronCount);

            return ls;
        }



    }
}
