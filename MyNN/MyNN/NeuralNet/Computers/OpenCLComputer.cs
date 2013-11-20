using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using MyNN.NeuralNet.Structure;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.NeuralNet.Computers
{
    public class OpenCLComputer : IMultilayerComputer
    {
        private readonly MNNCLProvider _clProvider;

        private Kernel[] _kernels;

        public OpenCLComputer(MNNCLProvider clProvider)
        {
            _clProvider = clProvider;
            _kernels = new Kernel[_clProvider.Network.Layers.Length];

            //загружаем программу и параметры
            LoadProgram();
        }

        private void LoadProgram()
        {
            for (var layerIndex = 1; layerIndex < _clProvider.Network.Layers.Length; layerIndex++)
            {
                var activationFunction = _clProvider.Network.Layers[layerIndex].LayerActivationFunction.GetOpenCLActivationFunction("lastNET");

                var kernelSource = _kernelSource.Replace(
                    "<activationFunction_lastNET>",
                    activationFunction);

                _kernels[layerIndex] = _clProvider.CreateKernel(kernelSource, "ComputeLayerKernel");
            }
        }

        public List<float[]> ComputeOutput(List<float[]> inputVectors)
        {
            var resultVector = new List<float[]>();

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //сохраняем состояние скрытых слоев в объекты OpenCL
            _clProvider.UnpackHidden();

            foreach (var inputVector in inputVectors)
            {
                //прописываем значения во входные нейроны
                _clProvider.Network.Layers[0].Compute(inputVector);

                //записываем входные нейроны в объекты OpenCL
                _clProvider.UnpackInput();

                // Make sure we're done with everything that's been requested before
                _clProvider.QueueFinish();

                //выполняем просчет сети
                this.ExecuteComputation();

                //извлекаем из сети выходные значения (результат отклика)
                var r = _clProvider.PackOutput();

                //сохраняем
                resultVector.Add(r);

                //повторяем для других входных значений
            }

            //восстанавливаем состояние скрытых слоев в нейросеть из объектов OpenCL
            _clProvider.PackHidden();

            return resultVector;
        }


        public float[] ComputeOutput(float[] inputVector)
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //прописываем значения во входные нейроны
            _clProvider.Network.Layers[0].Compute(inputVector);

            //записываем входные нейроны в объекты OpenCL
            _clProvider.Unpack();

            ExecuteComputation();

            //восстанавливаем состояние сети из объектов OpenCL в объект сети
            var result = _clProvider.Pack();

            return result;
        }

        public void ExecuteComputation()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //начинаем считать
            var layerCount = _clProvider.Network.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                _kernels[layerIndex]
                    .SetKernelArgMem(0, _clProvider.NeuronMem[layerIndex - 1])
                    .SetKernelArgMem(1, _clProvider.NeuronMem[layerIndex])
                    .SetKernelArgMem(2, _clProvider.WeightMem[layerIndex])
                    .SetKernelArg(3, 4, _clProvider.Network.Layers[layerIndex - 1].Neurons.Length)
                    .EnqueueNDRangeKernel(_clProvider.Network.Layers[layerIndex].NonBiasNeuronCount);
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        private string _kernelSource = @"
typedef struct
{
    float LastNET;
    float LastState;
    float Dedz;
} Neuron;

//const __constant float _alpha = 0.2;
//const __constant float _beta = 1.0;


int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void
        ComputeLayerKernel(
            __global Neuron * previousLayerNeurons,
            __global Neuron * currentLayerNeurons,
            __global float * weights,
            int previousLayerNeuronCount)
{
    int neuronIndex = get_global_id(0);

    //compute LastNET

    int weightIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

    float lastNET = 0;
    for (int plnIndex = 0; plnIndex < previousLayerNeuronCount; ++plnIndex)
    {
        lastNET += weights[weightIndex++] * previousLayerNeurons[plnIndex].LastState;
    }

    currentLayerNeurons[neuronIndex].LastNET = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerNeurons[neuronIndex].LastState = lastState;
}
 ";
    }
}
