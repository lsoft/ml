using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.DropConnectBit.WeightMask;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.ForwardPropagation
{
    public class DropConnectBitOpenCLForwardPropagation : IForwardPropagation
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
        private readonly IOpenCLWeightBitMaskContainer _weightMask;

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

        public DropConnectBitOpenCLForwardPropagation(
            VectorizationSizeEnum vse,
            MLP mlp,
            CLProvider clProvider,
            IOpenCLWeightBitMaskContainer weightMask
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
            if (weightMask == null)
            {
                throw new ArgumentNullException("weightMask");
            }

            _vse = vse;
            _mlp = mlp;
            _clProvider = clProvider;
            _weightMask = weightMask;

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
            var layerCount = _mlp.Layers.Length;

            NetMem = new MemFloat[layerCount];
            StateMem = new MemFloat[layerCount];
            WeightMem = new MemFloat[layerCount];

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
            __global uint * mask,
            int previousLayerNeuronCount1,
            int previousLayerNeuronCount1M1,
            int previousLayerNeuronCountTotal,

            int bitmask)
{
    //оригинальный алгоритм более чем в два раза медленен

    int neuronIndex = get_global_id(0);

    //compute LastNET
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    float lastNET = 0;
    for (int plnIndex =0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        uint mask1i = mask[weightIndex];
        float mask1 = ((mask1i & bitmask) > 0) ? (float)1 : (float)0;

        lastNET += mask1 * weights[weightIndex] * previousLayerLastState[plnIndex];
        weightIndex++;
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}


__kernel void
        ComputeLayerKernel4(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            __global uint * mask,
            int previousLayerNeuronCount4,
            int previousLayerNeuronCount4M4,
            int previousLayerNeuronCountTotal,

            int bitmask)
{
    int neuronIndex = get_global_id(0);

    //compute LastNET

    //забираем векторизованные данные

    //смещение в массиве весов на первый элемент
    int beginWeightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float4 lastNET4 = 0;
    for (
        int plnIndex4 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex4 = beginWeightIndex / 4, //индексатор на первый элемент float4 в массиве весов
            weightShift4 = beginWeightIndex - weightIndex4 * 4; //смещение для получения правильного float4 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex4 < previousLayerNeuronCount4;
        ++plnIndex4, ++weightIndex4)
    {
        float4 weights4 = vload4(weightIndex4, weights + weightShift4);
        uint4 mask4i = vload4(weightIndex4, mask + weightShift4);
        float4 mask4 = convert_float4(((mask4i & bitmask) > 0) ? 1 : 0);

        float4 previousLayerLastState4 = vload4(plnIndex4, previousLayerLastState);

        lastNET4 += mask4 * weights4 * previousLayerLastState4;
    }

    float lastNET = lastNET4.s0 + lastNET4.s1 + lastNET4.s2 + lastNET4.s3;

    //добираем невекторизованные данные (максимум - 3 флоата)
    for (
        int plnIndex = previousLayerNeuronCount4M4, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount4M4; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        uint mask1i = mask[weightIndex];
        float mask1 = ((mask1i & bitmask) > 0) ? (float)1 : (float)0;

        lastNET += mask1 * weights[weightIndex] * previousLayerLastState[plnIndex];
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}

__kernel void
        ComputeLayerKernel16(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            __global uint * mask,
            int previousLayerNeuronCount16,
            int previousLayerNeuronCount16M16,
            int previousLayerNeuronCountTotal,

            int bitmask)
{
    int neuronIndex = get_global_id(0);

    //compute LastNET

    //забираем векторизованные данные

    //смещение в массиве весов на первый элемент
    int beginWeightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float16 lastNET16 = 0;
    for (
        int plnIndex16 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex16 = beginWeightIndex / 16, //индексатор на первый элемент float16 в массиве весов
            weightShift16 = beginWeightIndex - weightIndex16 * 16; //смещение для получения правильного float16 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex16 < previousLayerNeuronCount16;
        ++plnIndex16, ++weightIndex16)
    {
        float16 weights16 = vload16(weightIndex16, weights + weightShift16);
        uint16 mask16i = vload16(weightIndex16, mask + weightShift16);
        float16 mask16 = convert_float16(((mask16i & bitmask) > 0) ? 1 : 0);

        float16 previousLayerLastState16 = vload16(plnIndex16, previousLayerLastState);

        lastNET16 += mask16 * weights16 * previousLayerLastState16;
    }

    float lastNET = 
          lastNET16.s0 
        + lastNET16.s1 
        + lastNET16.s2 
        + lastNET16.s3
        + lastNET16.s4
        + lastNET16.s5
        + lastNET16.s6
        + lastNET16.s7
        + lastNET16.s8
        + lastNET16.s9
        + lastNET16.sa
        + lastNET16.sb
        + lastNET16.sc
        + lastNET16.sd
        + lastNET16.se
        + lastNET16.sf
        ;

    //добираем невекторизованные данные (максимум - 15 флоатов)
    for (
        int plnIndex = previousLayerNeuronCount16M16, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount16M16; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        uint mask1i = mask[weightIndex];
        float mask1 = ((mask1i & bitmask) > 0) ? (float)1 : (float)0;

        lastNET += mask1 * weights[weightIndex] * previousLayerLastState[plnIndex];
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
            var result = new List<ILayerState>();

            this.PushWeights();

            this.ClearAndPushHiddenLayers();

            foreach (var d in dataSet)
            {
                this.Propagate(d);

                this.PopLastLayerState();

                var ls = new LayerState(this._lastLayerStateMem.Array, this.MLP.Layers.Last().NonBiasNeuronCount);
                result.Add(ls);
            }

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

                var vectorizationSize = VectorizationHelper.GetVectorizationSize(_vse);

                _kernels[layerIndex]
                    .SetKernelArgMem(0, this.StateMem[layerIndex - 1])
                    .SetKernelArgMem(1, this.NetMem[layerIndex])
                    .SetKernelArgMem(2, this.StateMem[layerIndex])
                    .SetKernelArgMem(3, this.WeightMem[layerIndex])
                    .SetKernelArgMem(4, this._weightMask.MaskMem[layerIndex])
                    .SetKernelArg(5, 4, prevLayerNeuronTotalCount / vectorizationSize)
                    .SetKernelArg(6, 4, prevLayerNeuronTotalCount - prevLayerNeuronTotalCount % vectorizationSize)
                    .SetKernelArg(7, 4, prevLayerNeuronTotalCount)
                    .SetKernelArg(8, 4, this._weightMask.BitMask)
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