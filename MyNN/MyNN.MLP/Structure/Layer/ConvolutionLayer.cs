using System;
using System.Linq;
using System.Net.Cache;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    [Serializable]
    public class ConvolutionLayer : IConvolutionLayer
    {
        private readonly float[] _kernel;
        private readonly float[] _biases;

        public IDimension KernelSpatialDimension
        {
            get;
            private set;
        }

        public LayerTypeEnum Type
        {
            get
            {
                return
                    LayerTypeEnum.Convolution;
            }
        }

        public IDimension SpatialDimension
        {
            get;
            private set;
        }

        /// <summary>
        /// Всего нейронов в слое
        /// </summary>
        public int TotalNeuronCount
        {
            get
            {
                return
                    this.FeatureMapCount * this.SpatialDimension.Multiplied;
            }
        }

        public INeuron[] Neurons
        {
            get;
            private set;
        }

        public IFunction LayerActivationFunction
        {
            get;
            private set;
        }

        public int FeatureMapCount
        {
            get
            {
                return
                    this.SpatialDimension.LastDimensionSize;
            }
        }

        public ConvolutionLayer(
            IRandomizer randomizer,
            INeuronFactory neuronFactory,
            IFunction activationFunction,
            IDimension spatialDimension,
            IDimension kernelSpatialDimension
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (kernelSpatialDimension == null)
            {
                throw new ArgumentNullException("kernelSpatialDimension");
            }

            LayerActivationFunction = activationFunction;
            SpatialDimension = spatialDimension;
            KernelSpatialDimension = kernelSpatialDimension;

            _kernel = new float[FeatureMapCount * KernelSpatialDimension.Multiplied];
            _kernel.Fill(j => randomizer.Next() - 0.5f);
            _biases = new float[FeatureMapCount];
            _biases.Fill(j => randomizer.Next() - 0.5f);

            this.Neurons = new INeuron[FeatureMapCount * SpatialDimension.Multiplied];

            for (var ni = 0; ni < this.Neurons.Length; ni++)
            {
                this.Neurons[ni] = neuronFactory.CreatePseudoNeuron();
            }
        }

        public string GetLayerInformation()
        {
            return
                string.Format(
                    "Conv(FM {3} x {0} K{2} {1})",
                    this.SpatialDimension.GetDimensionInformation(),
                    this.LayerActivationFunction.ShortName,
                    this.KernelSpatialDimension.GetDimensionInformation(),
                    this.FeatureMapCount
                    );
        }

        public ILayerConfiguration GetConfiguration()
        {
            return
                new LayerConfiguration(
                    this.LayerActivationFunction,
                    this.SpatialDimension,
                    this._kernel.Length,
                    this._biases.Length,
                    this.Neurons.ConvertAll(j => j.GetConfiguration())
                    );
        }

        /// <summary>
        /// Получить массив клонированных весов всех нейронов сети
        /// </summary>
        public void GetClonedWeights(
            out float[] weights,
            out float[] biases
            )
        {
            weights = _kernel.CloneArray();
            biases = _biases.CloneArray();
        }

        /// <summary>
        /// Записать веса в слой
        /// </summary>
        public void SetWeights(
            float[] weights,
            float[] biases
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (biases == null)
            {
                throw new ArgumentNullException("biases");
            }
            if (biases.Length != FeatureMapCount)
            {
                throw new ArgumentException("biases.Length != FeatureMapCount");
            }

            weights.CopyTo(_kernel, 0);
            biases.CopyTo(_biases, 0);
        }

    }
}