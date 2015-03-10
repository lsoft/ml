using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    [Serializable]
    public class AvgPoolingLayer : IAvgPoolingLayer
    {
        public int InverseScaleFactor
        {
            get
            {
                var fisf = 1f / this.ScaleFactor;

                if (Math.Abs(fisf - (int)fisf) > float.Epsilon)
                {
                    throw new ArgumentException("Invalid scale factor");
                }

                var isf = (int)fisf;

                return isf;
            }
        }

        public float ScaleFactor
        {
            get;
            private set;
        }

        public LayerTypeEnum Type
        {
            get
            {
                return
                    LayerTypeEnum.AvgPool;
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
            get
            {
                throw new NotSupportedException("Не поддерживается для пулинг-слоя");
            }
        }

        public int FeatureMapCount
        {
            get
            {
                return
                    this.SpatialDimension.LastDimensionSize;
            }
        }

        public AvgPoolingLayer(
            INeuronFactory neuronFactory,
            IDimension spatialDimension,
            float scaleFactor
            )
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (scaleFactor <= 0f || scaleFactor > 0.5f)
            {
                throw new ArgumentException("scaleFactor <= 0f || scaleFactor > 0.5f");
            }

            SpatialDimension = spatialDimension;
            ScaleFactor = scaleFactor;

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
                    "AvgPool(FM {1} x {0})",
                    this.SpatialDimension.GetDimensionInformation(),
                    this.FeatureMapCount
                    );
        }

        public ILayerConfiguration GetConfiguration()
        {
            return
                new LayerConfiguration(
                    this.LayerActivationFunction,
                    this.SpatialDimension,
                    0,
                    0,
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
            weights = new float[0];
            biases = new float[0];
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
            if (weights.Length != 0)
            {
                throw new ArgumentException("weights.Length != 0");
            }
            if (biases.Length != 0)
            {
                throw new ArgumentException("biases.Length != 0");
            }

            //nothing to do
        }

    }
}