using System;
using System.Collections.Generic;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Common.Other;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm
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

        public void ExecuteGibbsSampling(
            int indexIntoBatch,
            int maxGibbsChainLength)
        {
            //sample hidden
            _calculator.SampleHidden(
                _container.Weights,
                _container.HiddenBiases,
                _container.Hidden0,
                _container.Input
                );

            for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
            {
                var ifFirst = cdi == 0;
                var ifLast = cdi == (maxGibbsChainLength - 1);

                //compute visible
                var mem = ifFirst ? _container.Hidden0 : _container.Hidden1;
                _calculator.CalculateVisible(
                    _container.Weights,
                    _container.VisibleBiases,
                    _container.Visible,
                    mem
                    );

                if (ifLast)
                {
                    //compute hidden
                    _calculator.CalculateHidden(
                        _container.Weights,
                        _container.HiddenBiases,
                        _container.Hidden1,
                        _container.Visible);
                }
                else
                {
                    //sample hidden
                    _calculator.SampleHidden(
                        _container.Weights,
                        _container.HiddenBiases,
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

        public float[] CalculateVisible()
        {
            _calculator.CalculateVisible(
                _container.Weights,
                _container.VisibleBiases,
                _container.Input,
                _container.Hidden0);

            var result = _container.Input.CloneArray();

            return result;
        }

        public float[] CalculateHidden()
        {
            _calculator.CalculateHidden(
                _container.Weights,
                _container.HiddenBiases,
                _container.Hidden0,
                _container.Input);

            var result = _container.Hidden0.CloneArray();

            return result;
        }

        public float[] CalculateReconstructed()
        {
            _calculator.CalculateHidden(
                _container.Weights,
                _container.HiddenBiases,
                _container.Hidden0,
                _container.Input);

            _calculator.CalculateVisible(
                _container.Weights,
                _container.VisibleBiases,
                _container.Visible,
                _container.Hidden0);

            var result = _container.Visible.CloneArray();

            return result;
        }

        public ICollection<float[]> GetFeatures()
        {
            var result = new List<float[]>();

            var h = new float[_container.Hidden0.Length];

            for (var cc = 0; cc < _container.HiddenNeuronCount; cc++)
            {
                h[cc] = 1;

                var feature = new float[_container.VisibleNeuronCount];

                _calculator.CalculateVisible(
                    _container.Weights,
                    _container.VisibleBiases,
                    feature,
                    h
                    );

                result.Add(feature.CloneArray());

                h[cc] = 0;
            }

            return result;
        }
    }
}