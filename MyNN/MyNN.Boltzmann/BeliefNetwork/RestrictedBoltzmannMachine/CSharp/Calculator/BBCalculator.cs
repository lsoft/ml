using System;
using System.Threading.Tasks;
using MyNN.Common.Randomizer;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator
{
    public class BBCalculator : ICalculator
    {
        private readonly IRandomizer _randomizer;
        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;

        public string VisibleFunctionName
        {
            get
            {
                return
                    "Binary";
            }
        }

        public string HiddenFunctionName
        {
            get
            {
                return
                    "Binary";
            }
        }


        public BBCalculator(
            IRandomizer randomizer,
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _randomizer = randomizer;
            _visibleNeuronCount = visibleNeuronCount;
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public void SampleHidden(
            float[] weights,
            float[] hiddenBiases,
            float[] targetHidden,
            float[] fromVisible
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (targetHidden == null)
            {
                throw new ArgumentNullException("targetHidden");
            }
            if (fromVisible == null)
            {
                throw new ArgumentNullException("fromVisible");
            }

            Parallel.For(0, _hiddenNeuronCount, hiddenIndex =>
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
                {
                    sum += weights[CalculateWeightIndex(hiddenIndex, visibleIndex)] * fromVisible[visibleIndex];
                }
                sum += hiddenBiases[hiddenIndex];

                //уникальный рандом 
                var random = _randomizer.Next();

                //вероятностное состояние нейрона
                var probability = ComputeSigmoid(sum);
                targetHidden[hiddenIndex] = (random <= probability ? 1f : 0f);
            }
            );//Parallel.For        
        }

        public void CalculateHidden(
            float[] weights, 
            float[] hiddenBiases,
            float[] targetHidden,
            float[] fromVisible
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (targetHidden == null)
            {
                throw new ArgumentNullException("targetHidden");
            }
            if (fromVisible == null)
            {
                throw new ArgumentNullException("fromVisible");
            }

            Parallel.For(0, _hiddenNeuronCount, hiddenIndex =>
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
                {
                    sum += weights[CalculateWeightIndex(hiddenIndex, visibleIndex)] * fromVisible[visibleIndex];
                }

                sum += hiddenBiases[hiddenIndex];

                //вероятностное состояние нейрона
                var probability = ComputeSigmoid(sum);
                targetHidden[hiddenIndex] = probability;
            }
            );//Parallel.For
        }

        public void SampleVisible(
            float[] weights,
            float[] visibleBiases,
            float[] targetVisible,
            float[] fromHidden
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (targetVisible == null)
            {
                throw new ArgumentNullException("targetVisible");
            }
            if (fromHidden == null)
            {
                throw new ArgumentNullException("fromHidden");
            }

            Parallel.For(0, _visibleNeuronCount, visibleIndex =>
            //for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
                {
                    sum +=
                        weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }

                sum += visibleBiases[visibleIndex];

                //уникальный рандом 
                var random = _randomizer.Next();

                //вероятностное состояние нейрона
                var probability = ComputeSigmoid(sum);
                targetVisible[visibleIndex] = (random <= probability ? 1f : 0f);
            }
            );//Parallel.For
        }

        public void CalculateVisible(
            float[] weights, 
            float[] visibleBiases,
            float[] targetVisible,
            float[] fromHidden
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (targetVisible == null)
            {
                throw new ArgumentNullException("targetVisible");
            }
            if (fromHidden == null)
            {
                throw new ArgumentNullException("fromHidden");
            }

            Parallel.For(0, _visibleNeuronCount, visibleIndex =>
            //for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
                {
                    sum +=
                        weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }
                sum += visibleBiases[visibleIndex];

                //вероятностное состояние нейрона
                var probability = ComputeSigmoid(sum);
                targetVisible[visibleIndex] = probability;
            }
            );//Parallel.For
        }

        private float ComputeSigmoid(float x)
        {
            var r = (float)(1.0 / (1.0 + Math.Exp(-x)));
            return r;
        }


        private int CalculateWeightIndex(
            int hiddenIndex,
            int visibleIndex
            )
        {
            return
                hiddenIndex * _visibleNeuronCount + visibleIndex;
        }
    }
}