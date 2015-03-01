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
        private readonly ILayerConfiguration _currentLayerConfiguration;
        private readonly ILearningAlgorithmConfig _config;

        public HiddenLayerKernel(
            ILayerConfiguration currentLayerConfiguration,
            ILearningAlgorithmConfig config
            )
        {
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            _currentLayerConfiguration = currentLayerConfiguration;
            _config = config;
        }

        public void CalculateOverwrite(
            float[] currentLayerNET,

            float[] previousLayerLastState,
            float[] currentLayerDeDz,

            float[] currentLayerWeights,
            
            float[] nabla,

            float[] dedyMem,

            int previousLayerNeuronCount,
            int currentLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount,

            float[] currentLayerBias,
            float[] nablaBias
            )
        {
            Parallel.For(0, currentLayerNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount; neuronIndex++)
            {
            //    //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)
            //    var accDeDy = new KahanAlgorithm.Accumulator();
            //    for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
            //    {
            //        int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount, nextNeuronIndex) + neuronIndex; //не векторизуется:(

            //        float w = nextLayerWeights[nextWeightIndex];
            //        float dedz = nextLayerDeDz[nextNeuronIndex];
            //        float dedy = w * dedz;

            //        KahanAlgorithm.AddElement(ref accDeDy, dedy);
            //    }

            //    float currentDeDz = accDeDy.Sum;

                float dedy = dedyMem[neuronIndex];

                float nOut = currentLayerNET[neuronIndex];
                var currentDeDz = dedy * _currentLayerConfiguration.LayerActivationFunction.ComputeFirstDerivative(nOut);
                currentLayerDeDz[neuronIndex] = currentDeDz;

                int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

                for (
                    int currentWeightIndex = 0; 
                    currentWeightIndex < previousLayerNeuronCount; 
                    ++currentWeightIndex)
                {
                    float prevOut = previousLayerLastState[currentWeightIndex];

                    float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
                    float coef = prevOut + regularizationCoef;
                    float deltaWeight = learningRate * currentDeDz * coef;

                    nabla[currentNablaIndex + currentWeightIndex] = deltaWeight;
                }

                float deltaBias =
                    learningRate *
                    currentDeDz *
                    (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

                nablaBias[neuronIndex] = deltaBias;

            }
            ); // Parallel.For
        }

        public void CalculateIncrement(
            float[] currentLayerNET,

            float[] previousLayerLastState,
            float[] currentLayerDeDz,

            float[] currentLayerWeights,

            float[] nabla,

            float[] dedyMem,

            int previousLayerNeuronCount,
            int currentLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount,

            float[] currentLayerBias,
            float[] nablaBias
            )
        {
            Parallel.For(0, currentLayerNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount; neuronIndex++)
            {
                ////просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)
                //var accDeDy = new KahanAlgorithm.Accumulator();
                //for (int nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronCount; ++nextNeuronIndex)
                //{
                //    int nextWeightIndex = ComputeWeightIndex(currentLayerNeuronCount, nextNeuronIndex) + neuronIndex; //не векторизуется:(

                //    float w = nextLayerWeights[nextWeightIndex];
                //    float dedz = nextLayerDeDz[nextNeuronIndex];
                //    float dedy = w * dedz;

                //    KahanAlgorithm.AddElement(ref accDeDy, dedy);
                //}

                //float currentDeDz = accDeDy.Sum;

                float dedy = dedyMem[neuronIndex];


                float nOut = currentLayerNET[neuronIndex];
                var currentDeDz = dedy * _currentLayerConfiguration.LayerActivationFunction.ComputeFirstDerivative(nOut);
                currentLayerDeDz[neuronIndex] = currentDeDz;

                int currentNablaIndex = ComputeWeightIndex(previousLayerNeuronCount, neuronIndex);

                for (
                    int currentWeightIndex = 0;
                    currentWeightIndex < previousLayerNeuronCount;
                    ++currentWeightIndex)
                {
                    float prevOut = previousLayerLastState[currentWeightIndex];

                    float regularizationCoef = regularizationFactor * currentLayerWeights[currentWeightIndex] / dataCount;
                    float coef = prevOut + regularizationCoef;
                    float deltaWeight = learningRate * currentDeDz * coef;

                    nabla[currentNablaIndex + currentWeightIndex] += deltaWeight;
                }

                float deltaBias =
                    learningRate *
                    currentDeDz *
                    (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

                nablaBias[neuronIndex] += deltaBias;

            }
            ); // Parallel.For
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
