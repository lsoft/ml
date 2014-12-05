using System;
using System.Collections.Generic;
using System.Linq;
using System.Resources;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp.Kernel
{
    internal class OutputLayerKernel
    {
        private readonly ILayer _layer;
        private readonly ILearningAlgorithmConfig _config;

        public OutputLayerKernel(
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

            float[] desiredOutput,

            float[] currentLayerWeights,
            
            float[] nabla,

            int previousLayerNeuronCountTotal,
            int currentLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount
            )
        {
            for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount ; neuronIndex++)
            {
                float nOut = currentLayerNET[neuronIndex];
                float deri = _layer.LayerActivationFunction.ComputeFirstDerivative(nOut);

                float metric = _config.TargetMetrics.CalculatePartialDerivativeByV2Index(
                    currentLayerLastState,
                    desiredOutput,
                    neuronIndex
                    );

                float n = deri*metric;

                currentLayerDeDz[neuronIndex] = n;

                int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                for (
                    int weightIndex = 0; 
                    weightIndex < previousLayerNeuronCountTotal; 
                    ++weightIndex)
                {
                    float deltaWeight =
                        learningRate *
                        n *
                        (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

                    nabla[nablaNeuronShift + weightIndex] = deltaWeight;
                }
            }
        }

        public void CalculateIncrement(
            float[] currentLayerNET,

            float[] previousLayerLastState,
            float[] currentLayerLastState,
            float[] currentLayerDeDz,

            float[] desiredOutput,

            float[] currentLayerWeights,

            float[] nabla,

            int previousLayerNeuronCountTotal,
            int currentLayerNeuronCount,

            float learningRate,
            float regularizationFactor,
            float dataCount
            )
        {
            for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount; neuronIndex++)
            {
                float nOut = currentLayerNET[neuronIndex];
                float deri = _layer.LayerActivationFunction.ComputeFirstDerivative(nOut);

                float metric = _config.TargetMetrics.CalculatePartialDerivativeByV2Index(
                    currentLayerLastState,
                    desiredOutput,
                    neuronIndex
                    );

                float n = deri * metric;

                currentLayerDeDz[neuronIndex] = n;

                int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                for (
                    int weightIndex = 0;
                    weightIndex < previousLayerNeuronCountTotal;
                    ++weightIndex)
                {
                    float deltaWeight =
                        learningRate *
                        n *
                        (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

                    nabla[nablaNeuronShift + weightIndex] += deltaWeight;
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
