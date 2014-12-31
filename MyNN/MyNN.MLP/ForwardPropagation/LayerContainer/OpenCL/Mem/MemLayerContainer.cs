﻿using System;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem
{
    public class MemLayerContainer : IMemLayerContainer
    {
        private readonly CLProvider _clProvider;
        private readonly int _previousLayerTotalNeuronCount;
        private readonly int _currentLayerTotalNeuronCount;

        public MemFloat WeightMem
        {
            get;
            private set;
        }

        public MemFloat BiasMem
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
            int currentLayerTotalNeuronCount
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _clProvider = clProvider;
            _previousLayerTotalNeuronCount = 0;
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
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;
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
                currentLayerTotalNeuronCount*previousLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            weightMem.Write(BlockModeEnum.Blocking);

            WeightMem = weightMem;

            //биасы
            var biasMem = clProvider.CreateFloatMem(
                currentLayerTotalNeuronCount,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
            biasMem.Write(BlockModeEnum.Blocking);

            BiasMem = biasMem;
        }

        public void ClearAndPushNetAndState()
        {
            ClearNetAndState();

            PushNetAndState();
        }

        public void ClearNetAndState()
        {
            var nml = this.NetMem.Array.Length;
            Array.Clear(this.NetMem.Array, 0, nml);

            var sml = this.StateMem.Array.Length;
            Array.Clear(this.StateMem.Array, 0, sml);
        }


        public void PushNetAndState()
        {
            this.NetMem.Write(BlockModeEnum.NonBlocking);

            this.StateMem.Write(BlockModeEnum.NonBlocking);
        }

        public void ReadInput(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }
            if (data.Length != _currentLayerTotalNeuronCount)
            {
                throw new ArgumentException("data.Length != _currentLayerTotalNeuronCount");
            }

            //записываем значения из сети в объекты OpenCL
            for (var neuronIndex = 0; neuronIndex < _currentLayerTotalNeuronCount; neuronIndex++)
            {
                this.NetMem.Array[neuronIndex] = 0; //LastNET
                this.StateMem.Array[neuronIndex] = data[neuronIndex];
            }

            this.NetMem.Write(BlockModeEnum.NonBlocking);
            this.StateMem.Write(BlockModeEnum.NonBlocking);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        public void ReadWeightsFromLayer(ILayer layer)
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            var weightShift = 0;

            for (var neuronIndex = 0; neuronIndex < layer.TotalNeuronCount; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                Array.Copy(
                    neuron.Weights,
                    0,
                    WeightMem.Array,
                    weightShift,
                    neuron.Weights.Length);

                weightShift += neuron.Weights.Length;

                this.BiasMem.Array[neuronIndex] = neuron.Bias;
            }

            WeightMem.Write(BlockModeEnum.NonBlocking);
            BiasMem.Write(BlockModeEnum.NonBlocking);
        }

        public void PopWeights()
        {
            if (this.WeightMem != null)
            {
                this.WeightMem.Read(BlockModeEnum.Blocking);
                this.BiasMem.Read(BlockModeEnum.Blocking);
            }
        }

        public void WritebackWeightsToMLP(ILayer layer)
        {
            var weightLayer = this.WeightMem;

            var weightShiftIndex = 0;
            for (var neuronIndex = 0; neuronIndex < layer.TotalNeuronCount; ++neuronIndex)
            {
                var neuron = layer.Neurons[neuronIndex];

                var weightCount = neuron.Weights.Length;

                Array.Copy(weightLayer.Array, weightShiftIndex, neuron.Weights, 0, weightCount);
                weightShiftIndex += weightCount;

                neuron.Bias = this.BiasMem.Array[neuronIndex];
            }
        }

        public void PopNetAndState()
        {
            //извлекаем из Opencl последний слой
            this.NetMem.Read(BlockModeEnum.Blocking);
            this.StateMem.Read(BlockModeEnum.Blocking);
        }

        public ILayerState GetLayerState()
        {
            var ls = new LayerState(
                this.StateMem.Array,
                _currentLayerTotalNeuronCount);

            return ls;
        }



    }
}
