using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;

namespace MyNN.MLP.Structure.Layer.WeightBiasIniter
{
    public interface IWeightBiasIniter
    {
        void FillWeights(
            float[] weights
            );

        void FillBiases(
            float[] biases
            );

        float FillBias();
    }

    public class ConvolutionWeightBiasIniter : IWeightBiasIniter
    {
        private readonly IRandomizer _randomizer;
        private readonly IDimension _spatialDimension;
        private readonly int _featureMapCount;

        public ConvolutionWeightBiasIniter(
            IRandomizer randomizer,
            IDimension spatialDimension,
            int featureMapCount
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }

            _randomizer = randomizer;
            _spatialDimension = spatialDimension;
            _featureMapCount = featureMapCount;
        }

        public void FillWeights(
            float[] weights
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }

            weights.Fill(j => _randomizer.Next() * 0.4f - 0.2f);

            for (var fmi = 0; fmi < _featureMapCount; fmi++)
            {
                var w = new ReferencedSquareFloat(
                    _spatialDimension,
                    weights,
                    fmi*_spatialDimension.Multiplied
                    );

                var sign = _randomizer.Next() > 0.5f ? 1f : -1f;

                switch (fmi % 4)
                {
                    case 0:
                        //horizontal
                    {
                        var height = _randomizer.Next(_spatialDimension.Height);

                        for (var left = 0; left < _spatialDimension.Width; left++)
                        {
                            w.AddValueFromCoordSafely(left, height, sign * _randomizer.Next() * 0.8f);
                        }
                    }
                        break;
                    case 1:
                        //vertical
                    {
                        var left = _randomizer.Next(_spatialDimension.Width);

                        for (var height = 0; height < _spatialDimension.Height; height++)
                        {
                            w.AddValueFromCoordSafely(left, height, sign * _randomizer.Next() * 0.8f);
                        }
                    }
                        break;
                    case 2:
                        //diagonal from lt to rd
                    {
                        var border = Math.Min(_spatialDimension.Width, sign * _spatialDimension.Height);
                        for (var c = 0; c < border; c++)
                        {
                            w.AddValueFromCoordSafely(c, c, 1f);
                        }
                    }
                        break;
                    case 3:
                        //diagonal from ld to ru
                    {
                        var border = Math.Min(_spatialDimension.Width, _spatialDimension.Height);
                        for (var c = 0; c < border; c++)
                        {
                            w.AddValueFromCoordSafely(border - c - 1, c, sign * _randomizer.Next() * 0.8f);
                        }
                    }
                        break;
                }
            }
        }

        public void FillBiases(
            float[] biases
            )
        {
            if (biases == null)
            {
                throw new ArgumentNullException("biases");
            }

            biases.Fill(j => _randomizer.Next() - 0.5f);
        }

        public float FillBias()
        {
            return
                _randomizer.Next();
        }
    }

    public class RandomWeightBiasIniter : IWeightBiasIniter
    {
        private readonly IRandomizer _randomizer;

        public RandomWeightBiasIniter(
            IRandomizer randomizer
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
        }

        public void FillWeights(
            float[] weights
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }

            weights.Fill(j => _randomizer.Next() - 0.5f);
        }

        public void FillBiases(
            float[] biases
            )
        {
            if (biases == null)
            {
                throw new ArgumentNullException("biases");
            }

            biases.Fill(j => _randomizer.Next() - 0.5f);
        }

        public float FillBias()
        {
            return 
                _randomizer.Next();
        }
    }
}
