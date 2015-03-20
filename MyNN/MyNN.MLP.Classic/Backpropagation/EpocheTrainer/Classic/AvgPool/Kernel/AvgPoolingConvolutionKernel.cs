using System;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.AvgPool.Kernel
{
    public class AvgPoolingConvolutionKernel
    {
        private readonly IAvgPoolingLayerConfiguration _currentLayerConfiguration;

        public AvgPoolingConvolutionKernel(
            IAvgPoolingLayerConfiguration currentLayerConfiguration
            )
        {
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }

            _currentLayerConfiguration = currentLayerConfiguration;
        }

        public void Calculate(
            IReferencedSquareFloat currentLayerDeDz,
            IReferencedSquareFloat nextLayerDeDy
            )
        {
            if (nextLayerDeDy == null)
            {
                throw new ArgumentNullException("nextLayerDeDy");
            }
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }

            //вычисляем dE/dz текущего слоя

            for (var i = 0; i < _currentLayerConfiguration.SpatialDimension.Width; i++)
            {
                for (var j = 0; j < _currentLayerConfiguration.SpatialDimension.Height; j++)
                {
                    var dedy = nextLayerDeDy.GetValueFromCoordSafely(i, j);

                    var dedz = dedy*1;
                    //для avg pooling dedy тоже самое что и  dedz, так как нет функции активации
                    //(или можно сказать функция активации линейна и ее производная равна 1, что
                    //и показано для наглядности в формуле)

                    currentLayerDeDz.SetValueFromCoordSafely(i, j, dedz);
                }
            }
        }
    }
}