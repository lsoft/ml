using System;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm
{
    public class CD : IAlgorithm
    {
        private readonly ICalculator _calculator;
        private readonly FloatArrayContainer _container;

        public string Name
        {
            get
            {
                return
                    string.Format(
                        "Contrastive divergence ({0}-{1})",
                        _calculator.VisibleFunctionName,
                        _calculator.HiddenFunctionName);
            }
        }

        public CD(
            ICalculator calculator,
            FloatArrayContainer container
            )
        {
            if (calculator == null)
            {
                throw new ArgumentNullException("calculator");
            }
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }

            _calculator = calculator;
            _container = container;
        }

        public void PrepareTrain(int batchSize)
        {
            //nothing to do in CD
        }

        public void PrepareBatch()
        {
            //nothing to do in CD
        }

        public void CalculateSamples(
            int indexIntoBatch,
            int maxGibbsChainLength)
        {
            //sample hidden
            _calculator.SampleHidden(
                _container.Weights,
                _container.Hidden0,
                _container.Input
                );

            for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
            {
                var ifFirst = cdi == 0;
                var ifLast = cdi == (maxGibbsChainLength - 1);

                //compute visible
                _calculator.CalculateVisible(
                    _container.Weights,
                    _container.Visible,
                    ifFirst ? _container.Hidden0 : _container.Hidden1
                    );

                if (ifLast)
                {
                    //compute hidden
                    _calculator.CalculateHidden(
                        _container.Weights,
                        _container.Hidden1,
                        _container.Visible);
                }
                else
                {
                    //sample hidden
                    _calculator.SampleHidden(
                        _container.Weights,
                        _container.Hidden1,
                        _container.Visible
                        );
                }
            }
        }

        public void BatchFinished()
        {
            //nothing to do in CD
        }

        public float[] CalculateReconstructed()
        {
            _calculator.CalculateHidden(
                _container.Weights,
                _container.Hidden0,
                _container.Input);

            _calculator.CalculateVisible(
                _container.Weights,
                _container.Visible,
                _container.Hidden0);

            return
                _container.Visible;
        }
    }
}