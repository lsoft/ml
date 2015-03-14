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
        private readonly ILayerConfiguration _layerConfiguration;
        private readonly ILearningAlgorithmConfig _config;

        public OutputLayerKernel(
            ILayerConfiguration layerConfiguration,
            ILearningAlgorithmConfig config
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }

            _layerConfiguration = layerConfiguration;
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
            float dataCount,

            float[] currentLayerBias,
            float[] nablaBias
            )
        {
            Parallel.For(0, currentLayerNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount ; neuronIndex++)
            {
                float z = currentLayerNET[neuronIndex];
                float dydz = _layerConfiguration.LayerActivationFunction.ComputeFirstDerivative(z);

                float dedy = _config.TargetMetrics.CalculatePartialDerivativeByV2Index(
                    currentLayerLastState,
                    desiredOutput,
                    neuronIndex
                    );

                float dedz = dedy * dydz;

                currentLayerDeDz[neuronIndex] = dedz;

                int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                for (
                    int weightIndex = 0; 
                    weightIndex < previousLayerNeuronCountTotal; 
                    ++weightIndex)
                {
                    float deltaWeight =
                        learningRate *
                        dedz *
                        (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

                    nabla[nablaNeuronShift + weightIndex] = deltaWeight;
                }

                float deltaBias =
                    learningRate *
                    dedz *
                    (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

                nablaBias[neuronIndex] = deltaBias;
            }
            ); //Parallel.For
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
            float dataCount,

            float[] currentLayerBias,
            float[] nablaBias
            )
        {
            Parallel.For(0, currentLayerNeuronCount, neuronIndex =>
            //for (var neuronIndex = 0; neuronIndex < currentLayerNeuronCount ; neuronIndex++)
            {
                float z = currentLayerNET[neuronIndex];
                float dydz = _layerConfiguration.LayerActivationFunction.ComputeFirstDerivative(z);

                float dedy = _config.TargetMetrics.CalculatePartialDerivativeByV2Index(
                    currentLayerLastState,
                    desiredOutput,
                    neuronIndex
                    );

                float dedz = dedy * dydz;

                currentLayerDeDz[neuronIndex] = dedz;

                int nablaNeuronShift = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

                for (
                    int weightIndex = 0;
                    weightIndex < previousLayerNeuronCountTotal;
                    ++weightIndex)
                {
                    float deltaWeight =
                        learningRate *
                        dedz *
                        (previousLayerLastState[weightIndex] + regularizationFactor * currentLayerWeights[nablaNeuronShift + weightIndex] / dataCount);

                    nabla[nablaNeuronShift + weightIndex] += deltaWeight;
                }

                float deltaBias =
                    learningRate *
                    dedz *
                    (1 + regularizationFactor * currentLayerBias[neuronIndex] / dataCount);

                nablaBias[neuronIndex] += deltaBias;
            }
            ); //Parallel.For
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
