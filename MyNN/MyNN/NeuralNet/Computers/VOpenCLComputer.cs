using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.NeuralNet.Structure;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.NeuralNet.Computers
{
    public class VOpenCLComputer : IMultilayerComputer
    {
        private readonly VNNCLProvider _clProvider;

        private readonly Kernel[] _kernels;

        public VOpenCLComputer(
            VNNCLProvider clProvider,
            bool vectorized4)
        {
            _clProvider = clProvider;
            _kernels = new Kernel[_clProvider.Network.Layers.Length];

            //��������� ��������� � ���������
            LoadProgram(vectorized4);
        }

        private void LoadProgram(bool vectorized4)
        {
            for (var layerIndex = 1; layerIndex < _clProvider.Network.Layers.Length; layerIndex++)
            {
                var activationFunction = _clProvider.Network.Layers[layerIndex].LayerActivationFunction.GetOpenCLActivationFunction("lastNET");

                var kernelSource = _kernelSource.Replace(
                    "<activationFunction_lastNET>",
                    activationFunction);

                _kernels[layerIndex] = _clProvider.CreateKernel(
                    kernelSource,
                    vectorized4 ? "ComputeLayerKernel4" : "ComputeLayerKernel1");
            }
        }

        public List<float[]> ComputeOutput(List<float[]> inputVectors)
        {
            var resultVector = new List<float[]>();

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //��������� ��������� ������� ����� � ������� OpenCL
            _clProvider.UnpackHidden();

            foreach (var inputVector in inputVectors)
            {
                //����������� �������� �� ������� �������
                _clProvider.Network.Layers[0].Compute(inputVector);

                //���������� ������� ������� � ������� OpenCL
                _clProvider.UnpackInput();

                // Make sure we're done with everything that's been requested before
                _clProvider.QueueFinish();

                //��������� ������� ����
                this.ExecuteComputation();

                //��������� �� ���� �������� �������� (��������� �������)
                var r = _clProvider.PackOutput();

                //���������
                resultVector.Add(r);

                //��������� ��� ������ ������� ��������
            }

            //��������������� ��������� ������� ����� � ��������� �� �������� OpenCL
            _clProvider.PackHidden();

            return resultVector;
        }


        public float[] ComputeOutput(float[] inputVector)
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //����������� �������� �� ������� �������
            _clProvider.Network.Layers[0].Compute(inputVector);

            //���������� ������� ������� � ������� OpenCL
            _clProvider.Unpack();

            ExecuteComputation();

            //��������������� ��������� ���� �� �������� OpenCL � ������ ����
            var result = _clProvider.Pack();

            return result;
        }

        public void ExecuteComputation()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //�������� �������
            var layerCount = _clProvider.Network.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var prevLayerNeuronTotalCount = _clProvider.Network.Layers[layerIndex - 1].Neurons.Length;

                _kernels[layerIndex]
                    .SetKernelArgMem(0, _clProvider.LastStateMem[layerIndex - 1])
                    .SetKernelArgMem(1, _clProvider.LastNetMem[layerIndex])
                    .SetKernelArgMem(2, _clProvider.LastStateMem[layerIndex])
                    .SetKernelArgMem(3, _clProvider.WeightMem[layerIndex])
                    .SetKernelArg(4, 4, prevLayerNeuronTotalCount / 4)
                    .SetKernelArg(5, 4, prevLayerNeuronTotalCount - prevLayerNeuronTotalCount % 4)
                    .SetKernelArg(6, 4, prevLayerNeuronTotalCount)
                    .EnqueueNDRangeKernel(_clProvider.Network.Layers[layerIndex].NonBiasNeuronCount);
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

        private string _kernelSource = @"
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
        ComputeLayerKernel1(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            int previousLayerNeuronCount4,
            int previousLayerNeuronCount4M4,
            int previousLayerNeuronCountTotal)
{
    //������������ �������� ����� ��� � ��� ���� ��������

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

    //�������� ��������������� ������

    //�������� � ������� ����� �� ������ �������
    int beginWeightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float4 lastNET4 = 0;
    for (
        int plnIndex4 = 0, //���������� ������ ��������� �������� ����. ����
            weightIndex4 = beginWeightIndex / 4, //���������� �� ������ ������� float4 � ������� �����
            weightShift4 = beginWeightIndex - weightIndex4 * 4; //�������� ��� ��������� ����������� float4 (��� ��� ��������, ��� ������� 33 � �������� ����. ���� 127 �������� ����� 4191, �� ������ 4)
        plnIndex4 < previousLayerNeuronCount4;
        ++plnIndex4, ++weightIndex4)
    {
        float4 weights4 = vload4(weightIndex4, weights + weightShift4);
        float4 previousLayerLastState4 = vload4(plnIndex4, previousLayerLastState);

        lastNET4 += weights4 * previousLayerLastState4;
    }

    float lastNET = lastNET4.s0 + lastNET4.s1 + lastNET4.s2 + lastNET4.s3;

    //�������� ����������������� ������ (�������� - 3 ������)
    for (
        int plnIndex = previousLayerNeuronCount4M4, //���������� ������ ��������� �������� ����. ����
            weightIndex = beginWeightIndex + previousLayerNeuronCount4M4; //���������� �� ������ �����
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
    }
}
