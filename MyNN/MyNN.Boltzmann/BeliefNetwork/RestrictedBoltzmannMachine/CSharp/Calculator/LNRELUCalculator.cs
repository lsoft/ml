using System;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator
{
    public class LNRELUCalculator : ICalculator
    {
        private readonly Normal _gaussRandom = new Normal(0, 1);

        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;

        public string VisibleFunctionName
        {
            get
            {
                return
                    "Linear";
            }
        }

        public string HiddenFunctionName
        {
            get
            {
                return
                    "NRELU";
            }
        }


        public LNRELUCalculator(
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {

            _visibleNeuronCount = visibleNeuronCount + 1; //bias neuron
            _hiddenNeuronCount = hiddenNeuronCount + 1; //bias neuron
        }


        public void CalculateVisible(
            float[] weights,
            float[] targetVisible,
            float[] fromHidden
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (targetVisible == null)
            {
                throw new ArgumentNullException("targetVisible");
            }
            if (fromHidden == null)
            {
                throw new ArgumentNullException("fromHidden");
            }

            Parallel.For(0, _visibleNeuronCount - 1, visibleIndex => 
            //for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount - 1; visibleIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
                {
                    sum +=
                        weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }

                targetVisible[visibleIndex] = sum;
            }
            );//Parallel.For
        }

        public void SampleVisible(
            float[] weights, 
            float[] targetVisible,
            float[] fromHidden
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (targetVisible == null)
            {
                throw new ArgumentNullException("targetVisible");
            }
            if (fromHidden == null)
            {
                throw new ArgumentNullException("fromHidden");
            }

            Parallel.For(0, _visibleNeuronCount - 1, visibleIndex => 
            //for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount - 1; visibleIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
                {
                    sum +=
                        weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }

                targetVisible[visibleIndex] = sum;
            }
            );//Parallel.For
        }


        public void CalculateHidden(
            float[] weights,
            float[] targetHidden,
            float[] fromVisible
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (targetHidden == null)
            {
                throw new ArgumentNullException("targetHidden");
            }
            if (fromVisible == null)
            {
                throw new ArgumentNullException("fromVisible");
            }

            Parallel.For(0, _hiddenNeuronCount - 1, hiddenIndex => 
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount - 1; hiddenIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
                {
                    sum += weights[CalculateWeightIndex(hiddenIndex, visibleIndex)] * fromVisible[visibleIndex];
                }

                targetHidden[hiddenIndex] = sum;
            }
            );//Parallel.For
        }

        public void SampleHidden(
            float[] weights,
            float[] targetHidden,
            float[] fromVisible
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (targetHidden == null)
            {
                throw new ArgumentNullException("targetHidden");
            }
            if (fromVisible == null)
            {
                throw new ArgumentNullException("fromVisible");
            }

            Parallel.For(0, _hiddenNeuronCount - 1, hiddenIndex => 
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount - 1; hiddenIndex++)
            {
                //высчитываем состояние скрытого нейрона
                float sum = 0f;
                for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount; visibleIndex++)
                {
                    sum += weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]*fromVisible[visibleIndex];
                }
                
                var sampled = SampleHiddenNeuronWithNRelu(sum);
                targetHidden[hiddenIndex] = sampled;
            }
            );//Parallel.For
        }

        private float SampleHiddenNeuronWithNRelu(float x)
        {
            lock (_gaussRandom)
            {
                var stdDev = ComputeSigmoid(x);

                _gaussRandom.StdDev = stdDev;

                var normalNoise = (float) _gaussRandom.Sample();

                return
                    Math.Max(0f, x + normalNoise);
            }
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
                hiddenIndex*_visibleNeuronCount + visibleIndex;
        }
    }
}