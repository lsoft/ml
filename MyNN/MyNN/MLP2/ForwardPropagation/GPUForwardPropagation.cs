using System;
using System.Collections.Generic;
using System.Globalization;
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

        private Kernel[] _mulKernels;

        public GPUForwardPropagation(
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
            _mulKernels = new Kernel[_mlp.Layers.Length];

            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                var activationFunction = _mlp.Layers[layerIndex].LayerActivationFunction.GetOpenCLActivationFunction("lastNET");

                var mulKernelSource = _kernelSource.Replace(
                    "<activationFunction_lastNET>",
                    activationFunction);

                mulKernelSource = mulKernelSource.Replace(
                    "{CURRENT_LAYER_NEURON_COUNT}",
                    _mlp.Layers[layerIndex].NonBiasNeuronCount.ToString(CultureInfo.InvariantCulture));

                mulKernelSource = mulKernelSource.Replace(
                    "{PREVIOUS_LAYER_NEURON_COUNT}",
                    _mlp.Layers[layerIndex - 1].Neurons.Length.ToString(CultureInfo.InvariantCulture));

                var kernelName = "ComputeLayerKernel";

                _mulKernels[layerIndex] = _clProvider.CreateKernel(
                    mulKernelSource,
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

#define WARP_SIZE 32

__kernel void ComputeLayerKernel(
    const __global float* previousLayerLastState,
    __global float* currentLayerLastNET,
    __global float * currentLayerLastState,
    const __global float* weights,
    __local float* partialDotProduct
    )
{
    uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    uint height = {CURRENT_LAYER_NEURON_COUNT};

   // Each work-group computes multiple elements of W
   for (uint y = get_group_id(0); y < height; y += get_num_groups(0))
   {
      const __global float* row = weights + y * width;

      // Each work-item accumulates as many products as necessary
      // into private variable 'sum'
      float sum = 0;
      for (uint x = get_local_id(0); x < width; x += get_local_size(0))
         sum += row[x] * previousLayerLastState[x];

      // Each partial dot product is stored in shared memory
      partialDotProduct[get_local_id(0)] = sum;

      // Perform parallel reduction to add each work-item's
      // partial dot product together

      // Synchronize to make sure each work-item is done writing to
      // partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);

      // Thread local ID within a warp
      uint id = get_local_id(0) & (WARP_SIZE - 1); 

      // Each warp reduces 64 (default) consecutive elements
      float warpResult = 0.0f;
      if (get_local_id(0) < get_local_size(0)/2 )
      {
          volatile __local float* p = partialDotProduct + 2 * get_local_id(0) - id;
          p[0] += p[32];
          p[0] += p[16];
          p[0] += p[8];
          p[0] += p[4];
          p[0] += p[2];
          p[0] += p[1];
          warpResult = p[0];
      }

      // Synchronize to make sure each warp is done reading
      // partialDotProduct before it is overwritten in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // The first thread of each warp stores the result of the reduction
      // at the beginning of partialDotProduct
      if (id == 0)
         partialDotProduct[get_local_id(0) / WARP_SIZE] = warpResult;

      // Synchronize to make sure each warp is done writing to
      // partialDotProduct before it is read in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // Number of remaining elements after the first reduction
      uint size = get_local_size(0) / (2 * WARP_SIZE);

      // get_local_size(0) is less or equal to 512 on NVIDIA GPUs, so
      // only a single warp is needed for the following last reduction
      // step
      if (get_local_id(0) < size / 2)
      {
         volatile __local float* p = partialDotProduct + get_local_id(0);

         if (size >= 8)
            p[0] += p[4];
         if (size >= 4)
            p[0] += p[2];
         if (size >= 2)
            p[0] += p[1];
      }

      // Write the result of the reduction to global memory
      if (get_local_id(0) == 0)
      {
         float lastNET = partialDotProduct[0];
         currentLayerLastNET[y] = lastNET;

         //compute last state
         float lastState = <activationFunction_lastNET>;
         currentLayerLastState[y] = lastState;
      }

      // Synchronize to make sure the first work-item is done with
      // reading partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);
   }
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

            //// Make sure we're done with everything that's been requested before
            //_clProvider.QueueFinish();

            //начинаем считать
            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                //var previousLayerNeuronTotalCount = _mlp.Layers[layerIndex - 1].Neurons.Length;
                //var currentLayerNeuronCount = _mlp.Layers[layerIndex].NonBiasNeuronCount;

                const int szLocalWorkSize = 256;
                int szGlobalWorkSize = 64 * _clProvider.Parameters.NumComputeUnits * szLocalWorkSize;

                _mulKernels[layerIndex]
                    .SetKernelArgMem(0, this.StateMem[layerIndex - 1])
                    .SetKernelArgMem(1, this.NetMem[layerIndex])
                    .SetKernelArgMem(2, this.StateMem[layerIndex])
                    .SetKernelArgMem(3, this.WeightMem[layerIndex])
                    .SetKernelArgLocalMem(4, 4 * szLocalWorkSize)
                    .EnqueueNDRangeKernel(
                        new int[]
                        {
                            szGlobalWorkSize
                        }
                        , new int[]
                        {
                            szLocalWorkSize
                        }
                        );
            }

            //// Make sure we're done with everything that's been requested before
            //_clProvider.QueueFinish();
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

        //public void PushWeights()
        //{
        //    var layerCount = _mlp.Layers.Length;

        //    //веса оставшихся слоев
        //    for (var layerIndex = 1; layerIndex < layerCount; ++layerIndex)
        //    {
        //        var layer = _mlp.Layers[layerIndex];
        //        var weightMem = WeightMem[layerIndex];

        //        for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
        //        {
        //            var neuron = layer.Neurons[neuronIndex];

        //            for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
        //            {
        //                weightMem.Array[weightIndex * layer.NonBiasNeuronCount + neuronIndex] = neuron.Weights[weightIndex];
        //            }
        //        }

        //        weightMem.Write(BlockModeEnum.NonBlocking);
        //    }
        //}

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
