using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.ForwardPropagation
{
    public class GPUForwardPropagation : IForwardPropagation
    {
        private readonly VectorizationSizeEnum _vse;
        private readonly MLP _mlp;

        public MLP MLP
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

        public GPUForwardPropagation(
            VectorizationSizeEnum vse,
            MLP mlp,
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
                var currentLayerNeuronCount = _mlp.Layers[cc].Neurons.Length;

                var netMem = _clProvider.CreateFloatMem(
                    currentLayerNeuronCount,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
                netMem.Write(BlockModeEnum.Blocking);
                NetMem[cc] = netMem;

                var stateMem = _clProvider.CreateFloatMem(
                    currentLayerNeuronCount,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
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
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
                weightMem.Write(BlockModeEnum.Blocking);
                WeightMem[cc] = weightMem;
            }
        }

        private void LoadProgram()
        {
            _kernels = new Kernel[_mlp.Layers.Length];

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                var activationFunction = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLActivationFunction("lastNET");

                var kernelSource = _kernelSource.Replace(
                    "<activationFunction_lastNET>",
                    activationFunction);

                var kernelName = VectorizationHelper.GetKernelName("ComputeLayerKernel", _vse);

                _kernels[layerIndex] = _clProvider.CreateKernel(
                    kernelSource,
                    kernelName);
            }
        }

        private string _kernelSource = @"
int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void
        ComputeLayerKernel1(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            int previousLayerNeuronCountTotal)
{
    //оригинальный алгоритм более чем в два раза медленен

    int neuronIndex = get_global_id(0);
    int currentLayerNeuronCount = get_global_size(0);

    int weightIndex = 
        neuronIndex;
        //ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    //compute LastNET
    float lastNET = 0;
    for (int plnIndex =0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        lastNET += weights[weightIndex] * previousLayerLastState[plnIndex];

        weightIndex += currentLayerNeuronCount;
        //weightIndex ++;
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}
";



        #endregion

        public List<ILayerState> ComputeOutput(DataSet dataSet)
        {
            TimeSpan propagationTime;
            var result = ComputeOutput(
                dataSet,
                out propagationTime);

            return result;
        }

        public List<ILayerState> ComputeOutput(DataSet dataSet, out TimeSpan propagationTime)
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

        public List<IMLPState> ComputeState(DataSet dataSet)
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

                if (_vse != VectorizationSizeEnum.NoVectorization)
                {
                    throw new NotSupportedException();
                }

                var currentLayerNeuronCount = _mlp.Layers[layerIndex].NonBiasNeuronCount;

                var globalNeuronSize = currentLayerNeuronCount;
                var globalWeightSize = prevLayerNeuronTotalCount;

                const int WorkGroupSize0 = 32;
                const int WorkGroupSize1 = 32;

                var corrected_globalNeuronSize = globalNeuronSize;
                if (WorkGroupSize0 > 1 && corrected_globalNeuronSize > WorkGroupSize0)
                {
                    corrected_globalNeuronSize += (WorkGroupSize0 - globalNeuronSize % WorkGroupSize0);
                }

                var corrected_globalWeightSize = globalWeightSize;
                if (WorkGroupSize1 > 1 && corrected_globalWeightSize > WorkGroupSize1)
                {
                    corrected_globalWeightSize += (WorkGroupSize1 - globalWeightSize % WorkGroupSize1);
                }

                var localNeuronSize = WorkGroupSize0;
                var localWeightSize = WorkGroupSize1;

                var localNeuronCount = corrected_globalNeuronSize / localNeuronSize;
                var localWeightCount = corrected_globalWeightSize / localWeightSize;

                _kernels[layerIndex]
                    .SetKernelArgMem(0, this.StateMem[layerIndex - 1])
                    .SetKernelArgMem(1, this.NetMem[layerIndex])
                    .SetKernelArgMem(2, this.StateMem[layerIndex])
                    .SetKernelArgMem(3, this.WeightMem[layerIndex])
                    //.SetKernelArgLocalMem(4, 4 * localNeuronCount * localWeightCount)
                    .SetKernelArg(4, 4, prevLayerNeuronTotalCount)
                    .EnqueueNDRangeKernel(
                        currentLayerNeuronCount
                        //new int[]
                        //{
                        //    localNeuronSize,//corrected_globalNeuronSize,
                        //    localWeightSize,//corrected_globalWeightSize,
                        //}
                        //,new int[]
                        //{
                        //    localNeuronSize,
                        //    localWeightSize
                        //}
                        );
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
                var layer = _mlp.Layers[layerIndex];
                var weightMem = WeightMem[layerIndex];

                for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
                {
                    var neuron = layer.Neurons[neuronIndex];

                    for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        weightMem.Array[weightIndex * layer.NonBiasNeuronCount + neuronIndex] = neuron.Weights[weightIndex];
                    }
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