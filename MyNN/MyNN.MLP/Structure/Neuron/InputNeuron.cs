﻿using System;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    [Serializable]
    public class InputNeuron : INeuron
    {
        private readonly int _thisIndex;

        #region private class

        [Serializable]
        private class InputFunction : IFunction
        {
            public string ShortName
            {
                get
                {
                    return
                        "Input";
                }
            }

            public float Compute(float x)
            {
                return 1f;
            }

            public float ComputeFirstDerivative(float x)
            {
                throw new InvalidOperationException("Неприменимо");
            }

            public string GetOpenCLFirstDerivative(string varName)
            {
                throw new InvalidOperationException("Неприменимо");
            }

            public string GetOpenCLActivationFunction(string varName)
            {
                return
                    "(1.0)";
            }
        }

        #endregion

        public float[] Weights
        {
            get;
            private set;
        }

        public IFunction ActivationFunction
        {
            get;
            private set;
        }

        public bool IsBiasNeuron
        {
            get
            {
                return false;
            }
        }

        public INeuronConfiguration GetConfiguration()
        {
            return 
                new NeuronConfiguration(
                    this.Weights.Length,
                    this.IsBiasNeuron);
        }

        public InputNeuron(
            int thisIndex)
        {
            this._thisIndex = thisIndex;
            this.ActivationFunction = new InputFunction();

            //случайные веса
            this.Weights = new float[0];
        }
    }
}