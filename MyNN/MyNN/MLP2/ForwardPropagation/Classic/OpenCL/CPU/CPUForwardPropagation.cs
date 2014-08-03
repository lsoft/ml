using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU
{
    /// <summary>
    /// MLP Forward propagation implemented in CPU-oriented (Intel) OpenCL
    /// </summary>
    public class CPUForwardPropagation : IForwardPropagation
    {
        private readonly VectorizationSizeEnum _vse;
        private readonly IMLP _mlp;

        public IMLP MLP
        {
            get
            {
                return _mlp;
            }
        }

        private readonly CLProvider _clProvider;

        public MemFloat[] WeightMem;
        public MemFloat[] NetMem;
        public MemFloat[] StateMem;

        private MemFloat _lastLayerNetMem
        {
            get
            {
                return
                    this.NetMem.Last();
            }
        }

        private MemFloat _lastLayerStateMem
        {
            get
            {
                return
                    this.StateMem.Last();
            }
        }

        private Kernel[] _kernels;

        public CPUForwardPropagation(
            VectorizationSizeEnum vse,
            IMLP mlp,
            CLProvider clProvider
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _vse = vse;
            _mlp = mlp;
            _clProvider = clProvider;

            this.PrepareInfrastructure();
        }

        #region prepare infrastructure


        private void PrepareInfrastructure()
        {
            GenerateMems();

            //загружаем программу и параметры
            LoadProgram();

        }

        private void GenerateMems()
        {
            NetMem = new MemFloat[_mlp.Layers.Length];
            StateMem = new MemFloat[_mlp.Layers.Length];
            WeightMem = new MemFloat[_mlp.Layers.Length];

            var layerCount = _mlp.Layers.Length;

            //нейроны
            for (var cc = 0; cc < layerCount; cc++)
            {
                var currentLayerTotalNeuronCount = _mlp.Layers[cc].Neurons.Length;

                var netMem = _clProvider.CreateFloatMem(
                    currentLayerTotalNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                netMem.Write(BlockModeEnum.Blocking);
                NetMem[cc] = netMem;

                var stateMem = _clProvider.CreateFloatMem(
                    currentLayerTotalNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                stateMem.Write(BlockModeEnum.Blocking);
                StateMem[cc] = stateMem;
            }

            //веса
            for (var cc = 1; cc < layerCount; cc++)
            {
                var previousLayerNeuronCount = _mlp.Layers[cc - 1].Neurons.Length;
                var currentLayerNeuronCount = _mlp.Layers[cc].NonBiasNeuronCount;  //without bias neuron

                var weightMem = _clProvider.CreateFloatMem(
                    currentLayerNeuronCount * previousLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                weightMem.Write(BlockModeEnum.Blocking);
                WeightMem[cc] = weightMem;
            }
        }

        private void LoadProgram()
        {
            var ks = new CPUKernelSource();

            _kernels = new Kernel[_mlp.Layers.Length];

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                var activationFunction = _mlp.Layers[layerIndex].LayerActivationFunction;

                string kernelName;
                var kernelSource = ks.GetKernelSource(
                    _vse,
                    activationFunction,
                    out kernelName
                    );

                _kernels[layerIndex] = _clProvider.CreateKernel(
                    kernelSource,
                    kernelName);
            }
        }

        #endregion

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
            var result = new List<ILayerState>();

            this.PushWeights();

            this.ClearAndPushHiddenLayers();

            var before = DateTime.Now;

            foreach (var d in dataSet)
            {
                this.Propagate(d);

                this.PopLastLayerState();

                var ls = new LayerState(this._lastLayerStateMem.Array, this.MLP.Layers.Last().NonBiasNeuronCount);
                result.Add(ls);
            }

            var after = DateTime.Now;
            propagationTime = (after - before);

            return result;
        }

        public List<IMLPState> ComputeState(IDataSet dataSet)
        {
            var result = new List<IMLPState>();

            this.PushWeights();

            this.ClearAndPushHiddenLayers();

            foreach (var d in dataSet)
            {
                this.Propagate(d);

                this.PopState();

                var listls = new List<ILayerState>();

                for (var layerIndex = 0; layerIndex < _mlp.Layers.Count(); layerIndex++)
                {
                    var mem = this.StateMem[layerIndex];
                    var ls = new LayerState(mem.Array, this.MLP.Layers[layerIndex].NonBiasNeuronCount);
                    listls.Add(ls);
                }

                result.Add(
                    new MLPState(listls.ToArray()));
            }

            return result;
        }

        public void ClearAndPushHiddenLayers()
        {
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                var nm = NetMem[layerIndex];
                var nml = nm.Array.Length;
                Array.Clear(nm.Array, 0, nml);
                nm.Array[nml - 1] = 1f;
                nm.Write(BlockModeEnum.NonBlocking);

                var sm = StateMem[layerIndex];
                var sml = sm.Array.Length;
                Array.Clear(sm.Array, 0, sml);
                sm.Array[sml - 1] = 1f;
                sm.Write(BlockModeEnum.NonBlocking);
            }
        }

        public void Propagate(DataItem d)
        {
            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            PushInput(d);
            
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //начинаем считать
            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var prevLayerNeuronTotalCount = _mlp.Layers[layerIndex - 1].Neurons.Length;

                var vectorizationSize = VectorizationHelper.GetVectorizationSize(_vse);

                _kernels[layerIndex]
                    .SetKernelArgMem(0, this.StateMem[layerIndex - 1])
                    .SetKernelArgMem(1, this.NetMem[layerIndex])
                    .SetKernelArgMem(2, this.StateMem[layerIndex])
                    .SetKernelArgMem(3, this.WeightMem[layerIndex])
                    .SetKernelArg(4, 4, prevLayerNeuronTotalCount)
                    .EnqueueNDRangeKernel(_mlp.Layers[layerIndex].NonBiasNeuronCount);
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        public void PushWeights()
        {
            var layerCount = _mlp.Layers.Length;

            //веса оставшихся слоев
            for (var layerIndex = 1; layerIndex < layerCount; ++layerIndex)
            {
                var weightShift = 0;

                var layer = _mlp.Layers[layerIndex];
                var weightMem = WeightMem[layerIndex];
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
        }

        public void PopState()
        {
            this.PopHiddenState();

            this.PopLastLayerState();
        }

        private void PopHiddenState()
        {
            var layerCount = _mlp.Layers.Length;

            //пишем результат обратно в сеть
            for (var layerIndex = 1; layerIndex < layerCount - 1; layerIndex++)
            {
                //читаем его из opencl
                NetMem[layerIndex].Read(BlockModeEnum.Blocking);
                StateMem[layerIndex].Read(BlockModeEnum.Blocking);
            }
        }

        private void PopLastLayerState()
        {
            //извлекаем из Opencl последний слой
            _lastLayerNetMem.Read(BlockModeEnum.Blocking);
            _lastLayerStateMem.Read(BlockModeEnum.Blocking);
        }

        /// <summary>
        /// распаковывает значения из сети в массивы для opencl
        /// </summary>
        private void PushInput(DataItem d)
        {
            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            //записываем значения в сеть
            var firstLayer = _mlp.Layers[0];
            var firstLayerNeuronCount = firstLayer.Neurons.Length;

            //записываем значения из сети в объекты OpenCL
            for (var neuronIndex = 0; neuronIndex < firstLayerNeuronCount; neuronIndex++)
            {
                var isBiasNeuron = neuronIndex == (firstLayerNeuronCount - 1);

                NetMem[0].Array[neuronIndex] = 0; //LastNET
                StateMem[0].Array[neuronIndex] = 
                    isBiasNeuron 
                        ? 1.0f
                        : d.Input[neuronIndex];
            }

            NetMem[0].Write(BlockModeEnum.NonBlocking);
            StateMem[0].Write(BlockModeEnum.NonBlocking);
        }





    }
}