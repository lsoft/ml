using System;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DeDyAggregator
{
    public class CSharpAvgPoolingDeDyAggregator : ICSharpDeDyAggregator
    {
        private readonly IConvolutionLayerConfiguration _previousLayerConfiguration;
        private readonly IAvgPoolingLayerConfiguration _aggregateLayerConfiguration;
        
        private readonly int _featureMapCount;

        public float[] DeDz
        {
            get;
            private set;
        }

        public float[] DeDy
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get
            {
                return
                    _aggregateLayerConfiguration.TotalNeuronCount;
            }
        }

        public CSharpAvgPoolingDeDyAggregator(
            IConvolutionLayerConfiguration previousLayerConfiguration,
            IAvgPoolingLayerConfiguration aggregateLayerConfiguration
            )
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            if (aggregateLayerConfiguration == null)
            {
                throw new ArgumentNullException("aggregateLayerConfiguration");
            }
            if (previousLayerConfiguration.FeatureMapCount != aggregateLayerConfiguration.FeatureMapCount)
            {
                throw new ArgumentException("previousLayerConfiguration.FeatureMapCount != aggregateLayerConfiguration.FeatureMapCount");
            }

            _previousLayerConfiguration = previousLayerConfiguration;
            _aggregateLayerConfiguration = aggregateLayerConfiguration;
            
            _featureMapCount = aggregateLayerConfiguration.FeatureMapCount;

            this.DeDz = new float[aggregateLayerConfiguration.TotalNeuronCount];
            this.DeDy = new float[previousLayerConfiguration.TotalNeuronCount];
        }

        public void Aggregate(
            )
        {
            for (var fmi = 0; fmi < _featureMapCount; fmi++)
            {
                var dedyShift = fmi * _previousLayerConfiguration.SpatialDimension.Multiplied;

                var dedy = new ReferencedSquareFloat(
                    _previousLayerConfiguration.SpatialDimension,
                    this.DeDy,
                    dedyShift
                    );

                var dedzShift = fmi * _aggregateLayerConfiguration.SpatialDimension.Multiplied;

                var dedz = new ReferencedSquareFloat(
                    _aggregateLayerConfiguration.SpatialDimension,
                    this.DeDz,
                    dedzShift
                    );

                //вычисляем значение ошибки (dE/dz) апсемплингом с пулинг слоя
                //(реализация неэффективная! переделать!)
                //умножением на _nextLayerConfiguration.ScaleFactor может
                //несколько раз браться одно и то же значение из nextLayerDeDz
                //можно попробовать предрассчитать индексы и брать готовые
                for (var j = 0; j < dedy.Height; j++)
                {
                    for (var i = 0; i < dedy.Width; i++)
                    {
                        var jp = (int) (j*_aggregateLayerConfiguration.ScaleFactor);
                        var ip = (int) (i*_aggregateLayerConfiguration.ScaleFactor);

                        var v = dedz.GetValueFromCoordSafely(ip, jp);

                        dedy.SetValueFromCoordSafely(i, j, v);
                    }
                }
            }
        }

        public void ClearAndWrite(
            )
        {
            this.DeDy.Clear();
            this.DeDz.Clear();
        }

    }
}