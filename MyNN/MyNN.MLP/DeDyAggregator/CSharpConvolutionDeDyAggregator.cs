using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.Connector;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DeDyAggregator
{
    /// <summary>
    /// dedy аггрегатор для слоя свертки, перед которым стоит слой avg pooling
    /// (на самом деле, на данный момент, я думаю, что без разницы какой слой
    /// стоит перед данным, главное, чтобы он был двумерным)
    /// </summary>
    public class CSharpConvolutionDeDyAggregator : ICSharpDeDyAggregator
    {
        private readonly IAvgPoolingLayerConfiguration _previousLayerConfiguration;
        private readonly ICSharpConvolutionLayerContainer _aggregateLayerContainer;
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IConnector _connector;

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
                    _aggregateLayerContainer.Configuration.TotalNeuronCount;
            }
        }

        public CSharpConvolutionDeDyAggregator(
            IAvgPoolingLayerConfiguration previousLayerConfiguration,
            ICSharpConvolutionLayerContainer aggregateLayerContainer,
            ICSharpConvolutionCalculator convolutionCalculator,
            IConnector connector
            )
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            if (aggregateLayerContainer == null)
            {
                throw new ArgumentNullException("aggregateLayerContainer");
            }
            if (convolutionCalculator == null)
            {
                throw new ArgumentNullException("convolutionCalculator");
            }
            if (connector == null)
            {
                throw new ArgumentNullException("connector");
            }

            _previousLayerConfiguration = previousLayerConfiguration;
            _aggregateLayerContainer = aggregateLayerContainer;
            _convolutionCalculator = convolutionCalculator;
            _connector = connector;

            this.DeDz = new float[aggregateLayerContainer.Configuration.TotalNeuronCount];
            this.DeDy = new float[previousLayerConfiguration.TotalNeuronCount];
        }

        public void Aggregate(
            )
        {
            var processedDict = new ConcurrentDictionary<int, bool>();
            
            Parallel.For(0, _aggregateLayerContainer.Configuration.FeatureMapCount, currentFmi =>
            //for (var currentFmi = 0; currentFmi < _aggregateLayerContainer.Configuration.FeatureMapCount; currentFmi++)
            {
                //var kernelShift = currentFmi * _aggregateLayerContainer.Configuration.KernelSpatialDimension.Multiplied;
                //var biasShift = currentFmi;

                //var kernelBiasContainer = new ReferencedKernelBiasContainer(
                //    _aggregateLayerContainer.Configuration.KernelSpatialDimension, 
                //    _aggregateLayerContainer.WeightMem,
                //    kernelShift, 
                //    _aggregateLayerContainer.BiasMem,
                //    biasShift
                //    );

                var weightShift = currentFmi * _aggregateLayerContainer.Configuration.KernelSpatialDimension.Multiplied;

                var weights = new ReferencedSquareFloat(
                    _aggregateLayerContainer.Configuration.KernelSpatialDimension,
                    _aggregateLayerContainer.WeightMem,
                    weightShift
                    );

                var dedzShift = currentFmi * _aggregateLayerContainer.Configuration.SpatialDimension.Multiplied;

                var dedz = new ReferencedSquareFloat(
                    _aggregateLayerContainer.Configuration.SpatialDimension,
                    this.DeDz,
                    dedzShift
                    );

                //for (var previousFmi = 0; previousFmi < _previousLayerConfiguration.FeatureMapCount; previousFmi++)
                foreach (var previousFmi in _connector.GetPreviousFeatureMapIndexes(currentFmi))
                {
                    var previousDeDyShift = previousFmi*_previousLayerConfiguration.SpatialDimension.Multiplied;

                    var dedy = new ReferencedSquareFloat(
                        _previousLayerConfiguration.SpatialDimension,
                        this.DeDy,
                        previousDeDyShift
                        );

                    if(!processedDict.ContainsKey(previousFmi))
                    //if (currentFmi == 0)
                    {
                        _convolutionCalculator.CalculateBackConvolutionWithOverwrite(
                            weights, //kernelBiasContainer,
                            dedz,
                            dedy
                            );

                        processedDict.AddOrUpdate(
                            previousFmi,
                            true,
                            (a, b) => true
                            );
                    }
                    else
                    {
                        _convolutionCalculator.CalculateBackConvolutionWithIncrement(
                            weights, //kernelBiasContainer,
                            dedz,
                            dedy
                            );
                    }
                }
            }
            ); //Parallel.For
        }

        public void ClearAndWrite(
            )
        {
            this.DeDy.Clear();
            this.DeDz.Clear();
        }

    }
}