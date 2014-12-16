using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel
{
    internal class HiddenLayerKernel
    {
        private readonly ILayer _layer;
        private readonly ILearningAlgorithmConfig _config;

        public HiddenLayerKernel(
            ILayer layer,
            ILearningAlgorithmConfig config
            )
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            _layer = layer;
            _config = config;
        }

        public void CalculateOverwrite(
             float[] currentLayerNET,

             float[] previousLayerLastState,
             float[] currentLayerLastState,
             float[] currentLayerDeDz,
             float[] nextLayerDeDz,

             float[] currentLayerWeights,
             float[] nextLayerWeights,
            
             float[] nabla,

            int previousLayerNeuronCount,
            int currentLayerNeuronCount,
            int nextLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount
            )
        {
            for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount; neuronIndex++)
            {
                int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);


                //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)
                var accDeDz = new KahanAlgorithm.Accumulator();
                for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
                {
                    int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex; //не векторизуется:(

                    float nextWeight = nextLayerWeights[nextWeightIndex];
                    float nextNabla = nextLayerDeDz[nextNeuronIndex];
                    float multiplied = nextWeight * nextNabla;

                    KahanAlgorithm.AddElement(ref accDeDz, multiplied);
                }

                float currentDeDz = accDeDz.Sum;


                float nOut = currentLayerNET[neuronIndex];
                currentDeDz *= _layer.LayerActivationFunction.ComputeFirstDerivative(nOut);
                currentLayerDeDz[neuronIndex] = currentDeDz;

                for (
                    int currentWeightIndex = 0; 
                    currentWeightIndex < previousLayerNeuronCount; 
                    ++currentWeightIndex)
                {
                    float prevOut = previousLayerLastState[currentWeightIndex];

                    float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
                    float coef = prevOut + regularizationCoef;
                    float n = learningRate * currentDeDz * coef;

                    nabla[currentNablaIndex + currentWeightIndex] = n;
                }

            }

        }

        public void CalculateIncrement(
             float[] currentLayerNET,

             float[] previousLayerLastState,
             float[] currentLayerLastState,
             float[] currentLayerDeDz,
             float[] nextLayerDeDz,

             float[] currentLayerWeights,
             float[] nextLayerWeights,

             float[] nabla,

            int previousLayerNeuronCount,
            int currentLayerNeuronCount,
            int nextLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount
            )
        {
            for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount; neuronIndex++)
            {
                int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);


                //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)
                var accDeDz = new KahanAlgorithm.Accumulator();
                for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
                {
                    int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount + 1, nextNeuronIndex) + neuronIndex; //не векторизуется:(

                    float nextWeight = nextLayerWeights[nextWeightIndex];
                    float nextNabla = nextLayerDeDz[nextNeuronIndex];
                    float multiplied = nextWeight * nextNabla;

                    KahanAlgorithm.AddElement(ref accDeDz, multiplied);
                }

                float currentDeDz = accDeDz.Sum;


                float nOut = currentLayerNET[neuronIndex];
                currentDeDz *= _layer.LayerActivationFunction.ComputeFirstDerivative(nOut);
                currentLayerDeDz[neuronIndex] = currentDeDz;

                for (
                    int currentWeightIndex = 0;
                    currentWeightIndex < previousLayerNeuronCount;
                    ++currentWeightIndex)
                {
                    float prevOut = previousLayerLastState[currentWeightIndex];

                    float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
                    float coef = prevOut + regularizationCoef;
                    float n = learningRate * currentDeDz * coef;

                    nabla[currentNablaIndex + currentWeightIndex] += n;
                }

            }
        }

        private static int ComputeWeightIndex(
            int previousLayerNeuronCount,
            int neuronIndex)
        {
            return
                previousLayerNeuronCount * neuronIndex;
        }

    }
}
