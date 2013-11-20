using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.NeuralNet.Structure;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.NeuralNet
{
    public class MNNCLProvider : CLProvider
    {
        public readonly MultiLayerNeuralNetwork Network;

        public MemFloat[] NeuronMem;
        public MemFloat[] WeightMem;

        public MNNCLProvider(MultiLayerNeuralNetwork network)
        {
            Network = network;

            this.GenerateMems();
        }

        private void GenerateMems()
        {
            NeuronMem = new MemFloat[Network.Layers.Length];
            WeightMem = new MemFloat[Network.Layers.Length];

            var layerCount = Network.Layers.Length;

            //нейроны
            for (var cc = 0; cc < layerCount; cc++)
            {
                var currentLayerNeuronCount = Network.Layers[cc].Neurons.Length;

                NeuronMem[cc] = this.CreateFloatMem(
                    currentLayerNeuronCount * 3,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            }

            //веса
            for (var cc = 1; cc < layerCount; cc++)
            {
                var previousLayerNeuronCount = Network.Layers[cc - 1].Neurons.Length;
                var currentLayerNeuronCount = Network.Layers[cc].NonBiasNeuronCount;  //without bias neuron

                WeightMem[cc] = this.CreateFloatMem(
                    currentLayerNeuronCount * previousLayerNeuronCount,
                    Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            }
        }

        /// <summary>
        /// распаковывает значения из сети в массивы для opencl
        /// </summary>
        public void UnpackInput()
        {
            //записываем значения в сеть
            var firstLayer = Network.Layers[0];
            var firstLayerNeuronCount = firstLayer.Neurons.Length;

            //записываем значения из сети в объекты OpenCL
            var firstMem = NeuronMem[0];
            for (var neuronIndex = 0; neuronIndex < firstLayerNeuronCount; neuronIndex++)
            {
                firstMem.Array[neuronIndex * 3 + 0] = 0; //LastNET
                firstMem.Array[neuronIndex * 3 + 1] = firstLayer.Neurons[neuronIndex].LastState; //LastState
                firstMem.Array[neuronIndex * 3 + 2] = 0; //dedz
            }

            firstMem.Write(BlockModeEnum.Blocking);
        }

        /// <summary>
        /// распаковывает значения из сети в массивы для opencl
        /// </summary>
        public void UnpackHidden()
        {
            var layerCount = Network.Layers.Length;

            //нейроны оставшихся слоев
            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                var lastLayer = layerIndex == (layerCount - 1);


                var currentLayer = Network.Layers[layerIndex];
                var currentLayerNeuronCount = currentLayer.Neurons.Length;

                var neuronMem = NeuronMem[layerIndex];
                for (var dd = 0; dd < currentLayerNeuronCount; dd++)
                {
                    var lastNeuron = dd == (currentLayerNeuronCount - 1);
                    var isBiasNeuron = (!lastLayer && lastNeuron);

                    var neuron = currentLayer.Neurons[dd];

                    neuronMem.Array[dd * 3 + 0] = isBiasNeuron ? 0.0f : neuron.LastNET; //LastNET
                    neuronMem.Array[dd * 3 + 1] = isBiasNeuron ? 1.0f : neuron.LastState; //LastState, 1.0f для bias нейронов (для других нейронов, 1.0f потом будет перезатерта)
                    neuronMem.Array[dd * 3 + 2] = isBiasNeuron ? 0.0f : neuron.Dedz; //dedz
                }

                neuronMem.Write(BlockModeEnum.Blocking);
            }


            //веса оставшихся слоев
            for (var layerIndex = 1; layerIndex < layerCount; ++layerIndex)
            {
                var weightShift = 0;

                var layer = Network.Layers[layerIndex];
                var weightMem = WeightMem[layerIndex];
                for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
                {
                    var neuron = layer.Neurons[neuronIndex];

                    //foreach (var w in neuron.Weights)
                    //{
                    //    weightMem.Array[weightShift++] = w;
                    //}

                    Array.Copy(
                        neuron.Weights,
                        0,
                        weightMem.Array,
                        weightShift,
                        neuron.Weights.Length);

                    weightShift += neuron.Weights.Length;
                }

                weightMem.Write(BlockModeEnum.Blocking);
            }
        }

        /// <summary>
        /// распаковывает значения из сети в массивы для opencl
        /// </summary>
        public void Unpack()
        {
            this.UnpackInput();
            this.UnpackHidden();
        }

        /// <summary>
        /// упаковывает информацию из массовов opencl в сеть
        /// и возвращает состояние (LastState) нейронов выходного слоя
        /// </summary>
        /// <returns></returns>
        public float[] Pack()
        {
            PackHidden();

            return
                PackOutput();
        }


        /// <summary>
        /// упаковывает информацию из массовов opencl в сеть
        /// и возвращает состояние (LastState) нейронов выходного слоя
        /// </summary>
        /// <returns></returns>
        public void PackHidden()
        {
            var layerCount = Network.Layers.Length;

            //пишем результат обратно в сеть
            for (var layerIndex = 1; layerIndex < layerCount - 1; layerIndex++)
            {
                //читаем его из opencl
                NeuronMem[layerIndex].Read(BlockModeEnum.Blocking);

                var layer = Network.Layers[layerIndex];
                for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
                {
                    var neuron = layer.Neurons[neuronIndex];
                    neuron.SetState(
                        NeuronMem[layerIndex].Array[neuronIndex * 3 + 0],
                        NeuronMem[layerIndex].Array[neuronIndex * 3 + 1],
                        NeuronMem[layerIndex].Array[neuronIndex * 3 + 2]);
                }

            }
        }

        /// <summary>
        /// упаковывает информацию из массовов opencl в сеть
        /// и возвращает состояние (LastState) нейронов выходного слоя
        /// </summary>
        /// <returns></returns>
        public float[] PackOutput()
        {
            //распаковываем последний слой

            var lastLayerIndex = Network.Layers.Length - 1;

            //читаем его из opencl
            NeuronMem[lastLayerIndex].Read(BlockModeEnum.Blocking);

            var lastLayer = Network.Layers[lastLayerIndex];
            for (var neuronIndex = 0; neuronIndex < lastLayer.NonBiasNeuronCount; neuronIndex++)
            {
                var neuron = lastLayer.Neurons[neuronIndex];
                neuron.SetState(
                    NeuronMem[lastLayerIndex].Array[neuronIndex * 3 + 0],
                    NeuronMem[lastLayerIndex].Array[neuronIndex * 3 + 1],
                    NeuronMem[lastLayerIndex].Array[neuronIndex * 3 + 2]);
            }

            //возвращаем результат
            var lastLayerResult = NeuronMem.Last();
            var result = new float[Network.Layers.Last().NonBiasNeuronCount];
            
            for (var cc = 0; cc < result.Length; cc++)
            {
                result[cc] = lastLayerResult.Array[cc * 3 + 1]; //retrieve LastState only
            }

            return result;
        }


    }
}
