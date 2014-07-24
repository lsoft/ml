using System;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.NegativeSampler
{
    public class CD : INegativeSampler
    {
        private readonly ICalculator _calculator;
        private readonly float[] _weights;
        private readonly float[] _visible;
        private readonly float[] _hidden0;
        private readonly float[] _hidden1;

        public string Name
        {
            get
            {
                return
                    string.Format(
                        "Contrastive divergence ({0})",
                        _calculator.Name);
            }
        }

        public CD(
            ICalculator calculator,
            float[] weights,
            float[] visible,
            float[] hidden0,
            float[] hidden1
            )
        {
            if (calculator == null)
            {
                throw new ArgumentNullException("calculator");
            }
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visible == null)
            {
                throw new ArgumentNullException("visible");
            }
            if (hidden0 == null)
            {
                throw new ArgumentNullException("hidden0");
            }
            if (hidden1 == null)
            {
                throw new ArgumentNullException("hidden1");
            }

            _calculator = calculator;
            _weights = weights;
            _visible = visible;
            _hidden0 = hidden0;
            _hidden1 = hidden1;
        }

        public void PrepareTrain(int batchSize)
        {
            //nothing to do in CD
        }

        public void PrepareBatch()
        {
            //nothing to do in CD
        }

        public void CalculateNegativeSample(
            int indexIntoBatch,
            int maxGibbsChainLength)
        {
            for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
            {
                var ifFirst = cdi == 0;
                var ifLast = cdi == (maxGibbsChainLength - 1);

                //compute visible
                _calculator.CalculateVisible(
                    _weights,
                    _visible,
                    ifFirst ? _hidden0 : _hidden1
                    );

                if (ifLast)
                {
                    //compute hidden
                    _calculator.CalculateHidden(
                        _weights,
                        _hidden1,
                        _visible);
                }
                else
                {
                    //sample hidden
                    _calculator.SampleHidden(
                        _weights,
                        _hidden1,
                        _visible
                        );
                }
            }
        }

        public void BatchFinished()
        {
            //nothing to do in CD
        }
    }
}