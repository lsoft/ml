using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Layer;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.ForwardPropagation.DropConnect.Inference
{
    /// <summary>
    /// MLP inferencer dropconnect forward propagation implemented in CPU-oriented OpenCL
    /// </summary>
    /// <typeparam name="T">Type of concrete layer inferencer</typeparam>
    public class InferenceOpenCLForwardPropagation<T> : IForwardPropagation
        where T : ILayerInference
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
        private readonly IRandomizer _randomizer;
        private readonly int _sampleCount;
        private readonly float _p;

        private T[] _inferencers;

        private MemFloat[] _weightMem;
        private MemFloat[] _netMem;
        private MemFloat[] _stateMem;

        private MemFloat _lastLayerNetMem
        {
            get
            {
                return
                    this._netMem.Last();
            }
        }

        private MemFloat _lastLayerStateMem
        {
            get
            {
                return
                    this._stateMem.Last();
            }
        }

        private Kernel[] _kernels;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="vse">Mode of vectorization</param>
        /// <param name="mlp">Dropconnect MLP</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="randomizer">Random number provider (it needs for stochastic inference)</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public InferenceOpenCLForwardPropagation(
            VectorizationSizeEnum vse,
            IMLP mlp,
            CLProvider clProvider,
            IRandomizer randomizer,
            int sampleCount = 10000,
            float p = 0.5f
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
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (sampleCount <= 0)
            {
                throw new ArgumentOutOfRangeException("sampleCount");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            if (sampleCount < 1000 || sampleCount > 10000)
            {
                ConsoleAmbientContext.Console.WriteWarning(
                    "Sample count = {0}. Sample count typically lay in [1000; 10000].",
                    sampleCount);
            }

            _vse = vse;
            _mlp = mlp;
            _clProvider = clProvider;
            _randomizer = randomizer;
            _sampleCount = sampleCount;
            _p = p;

            this.PrepareInfrastructure();
        }

        #region prepare infrastructure


        private void PrepareInfrastructure()
        {
            GenerateMems();

            //загружаем программу и параметры
            LoadProgram();

            //create inferencers
            CreateInferencers();
        }

        private void CreateInferencers()
        {
            this._inferencers =new T[this._mlp.Layers.Length];

            for (var li = 1; li < this._mlp.Layers.Length; li++)
            {
                var prevLayer = this._mlp.Layers[li - 1];
                var currentLayer = this._mlp.Layers[li];

                var weightMem = this._weightMem[li];

                var prevLayerStateMem = this._stateMem[li - 1];
                var currentLayerStateMem = this._stateMem[li];

                var inferencer = (T)Activator.CreateInstance(
                    typeof(T),
                    _randomizer,
                    _clProvider,
                    _sampleCount,
                    prevLayer,
                    currentLayer,
                    weightMem,
                    prevLayerStateMem,
                    currentLayerStateMem,
                    _p
                    );

                this._inferencers[li] = inferencer;
            }
        }

        private void GenerateMems()
        {
            _netMem = new MemFloat[_mlp.Layers.Length];
            _stateMem = new MemFloat[_mlp.Layers.Length];
            _weightMem = new MemFloat[_mlp.Layers.Length];

            var layerCount = _mlp.Layers.Length;

            //нейроны
            for (var cc = 0; cc < layerCount; cc++)
            {
                var currentLayerNeuronCount = _mlp.Layers[cc].Neurons.Length;

                var netMem = _clProvider.CreateFloatMem(
                    currentLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                netMem.Write(BlockModeEnum.Blocking);
                _netMem[cc] = netMem;

                var stateMem = _clProvider.CreateFloatMem(
                    currentLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                stateMem.Write(BlockModeEnum.Blocking);
                _stateMem[cc] = stateMem;
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
                _weightMem[cc] = weightMem;
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

inline int ComputeWeightIndex(
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
            int previousLayerNeuronCount4,
            int previousLayerNeuronCount4M4,
            int previousLayerNeuronCountTotal)
{
    //оригинальный алгоритм более чем в два раза медленен

    int neuronIndex = get_global_id(0);

    //compute LastNET
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    float lastNET = 0;
    for (int plnIndex =0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        lastNET += weights[weightIndex++] * previousLayerLastState[plnIndex];
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
            int previousLayerNeuronCount4,
            int previousLayerNeuronCount4M4,
            int previousLayerNeuronCountTotal)
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
        float4 previousLayerLastState4 = vload4(plnIndex4, previousLayerLastState);

        lastNET4 += weights4 * previousLayerLastState4;
    }

    float lastNET = lastNET4.s0 + lastNET4.s1 + lastNET4.s2 + lastNET4.s3;

    //добираем невекторизованные данные (максимум - 3 флоата)
    for (
        int plnIndex = previousLayerNeuronCount4M4, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount4M4; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        lastNET += weights[weightIndex] * previousLayerLastState[plnIndex];
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
            int previousLayerNeuronCount16,
            int previousLayerNeuronCount16M16,
            int previousLayerNeuronCountTotal)
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
        float16 previousLayerLastState16 = vload16(plnIndex16, previousLayerLastState);

        lastNET16 += weights16 * previousLayerLastState16;
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
        lastNET += weights[weightIndex] * previousLayerLastState[plnIndex];
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}
";



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
                    var mem = this._stateMem[layerIndex];
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
                var nm = _netMem[layerIndex];
                var nml = nm.Array.Length;
                Array.Clear(nm.Array, 0, nml);
                nm.Array[nml - 1] = 1f;
                nm.Write(BlockModeEnum.NonBlocking);

                var sm = _stateMem[layerIndex];
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
            
            //// Make sure we're done with everything that's been requested before
            //_clProvider.QueueFinish();

            //начинаем считать
            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var prevLayerNeuronTotalCount = _mlp.Layers[layerIndex - 1].Neurons.Length;

                var vectorizationSize = VectorizationHelper.GetVectorizationSize(_vse);

                _kernels[layerIndex]
                    .SetKernelArgMem(0, this._stateMem[layerIndex - 1])
                    .SetKernelArgMem(1, this._netMem[layerIndex])
                    .SetKernelArgMem(2, this._stateMem[layerIndex])
                    .SetKernelArgMem(3, this._weightMem[layerIndex])
                    .SetKernelArg(4, 4, prevLayerNeuronTotalCount / vectorizationSize)
                    .SetKernelArg(5, 4, prevLayerNeuronTotalCount - prevLayerNeuronTotalCount % vectorizationSize)
                    .SetKernelArg(6, 4, prevLayerNeuronTotalCount)
                    .EnqueueNDRangeKernel(_mlp.Layers[layerIndex].NonBiasNeuronCount);

                _inferencers[layerIndex].InferenceLayer();
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
                var weightMem = _weightMem[layerIndex];
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
                _netMem[layerIndex].Read(BlockModeEnum.Blocking);
                _stateMem[layerIndex].Read(BlockModeEnum.Blocking);
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

                _netMem[0].Array[neuronIndex] = 0; //LastNET
                _stateMem[0].Array[neuronIndex] = 
                    isBiasNeuron 
                        ? 1.0f
                        : d.Input[neuronIndex];
            }

            _netMem[0].Write(BlockModeEnum.NonBlocking);
            _stateMem[0].Write(BlockModeEnum.NonBlocking);
        }





    }
}