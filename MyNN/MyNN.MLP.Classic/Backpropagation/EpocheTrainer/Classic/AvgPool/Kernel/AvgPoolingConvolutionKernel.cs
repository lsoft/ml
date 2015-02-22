using System;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel
{
    public class AvgPoolingConvolutionKernel
    {
        private readonly ICSharpConvolutionCalculator _convolutionCalculator;
        private readonly IAvgPoolingLayer _currentLayer;
        private readonly IConvolutionLayer _nextLayer;

        public AvgPoolingConvolutionKernel(
            ICSharpConvolutionCalculator convolutionCalculator,
            IAvgPoolingLayer currentLayer,
            IConvolutionLayer nextLayer
            )
        {
            if (convolutionCalculator == null)
            {
                throw new ArgumentNullException("convolutionCalculator");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (nextLayer == null)
            {
                throw new ArgumentNullException("nextLayer");
            }

            _convolutionCalculator = convolutionCalculator;
            _currentLayer = currentLayer;
            _nextLayer = nextLayer;
        }

        public void Calculate(
            int currentLayerFmiNeuronIndex,
            IReferencedSquareFloat currentLayerDeDz,
            float[] nextLayerDeDz,
            float[] nextLayerWeights
            )
        {
            if (nextLayerDeDz == null)
            {
                throw new ArgumentNullException("nextLayerDeDz");
            }
            if (nextLayerWeights == null)
            {
                throw new ArgumentNullException("nextLayerWeights");
            }
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }

            //вычисляем значение ошибки (dE/dz) суммированием по след слою
            //а след слой - сверточный, поэтому суммируем согласно ядру свертки

            for (var nextLayerFmi = 0; nextLayerFmi < _nextLayer.FeatureMapCount; nextLayerFmi++)
            {
                //вычисляем dE/dy текущего слоя

                var nextLayerShift = nextLayerFmi * _nextLayer.SpatialDimension.Multiplied;


                var nextLayerDeDzShifted = new ReferencedSquareFloat(
                    _nextLayer.SpatialDimension,
                    nextLayerDeDz,
                    nextLayerShift
                    );

                //var nextLayerWeightsShifted = new ReferencedSquareFloat(
                //    _nextLayer.SpatialDimension,
                //    nextLayerWeights,
                //    nextLayerShift
                //    );


                if (nextLayerFmi == 0)
                {
                    _convolutionCalculator.CalculateBackConvolutionWithOverwrite(
                        nextLayerDeDzShifted,
                        nextLayerWeights,
                        currentLayerDeDz
                        );
                }
                else
                {
                    _convolutionCalculator.CalculateBackConvolutionWithIncrement(
                        nextLayerDeDzShifted,
                        nextLayerWeights,
                        currentLayerDeDz
                        );
                }
            }

            //вычисляем dE/dz текущего слоя

            //для avg pooling dedy тоже самое что и dedz, так как нет функции активации
            //(или можно сказать функция активации линейна и ее производная равна 1)
            //поэтому она здесь не показана
        }
    }
}