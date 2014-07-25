using System;
using System.Xml;
using AForge;
using MyNN.BeliefNetwork.FreeEnergyCalculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.Data;
using MyNN.MLP2.ArtifactContainer;
using MyNN.Randomizer;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container
{
    public class FloatArrayContainer : IContainer
    {
        private readonly IFreeEnergyCalculator _freeEnergyCalculator;

        public float[] Input
        {
            get;
            private set;
        }

        public float[] Visible
        {
            get;
            private set;
        }

        public float[] Hidden0
        {
            get;
            private set;
        }

        public float[] Hidden1
        {
            get;
            private set;
        }

        public float[] Weights
        {
            get;
            private set;
        }

        private readonly float[] _nabla;

        public int VisibleNeuronCount
        {
            get;
            private set;
        }

        public int HiddenNeuronCount
        {
            get;
            private set;
        }

        private readonly int _visibleNeuronCountWithBias;
        private readonly int _hiddenNeuronCountWithBias;

        public FloatArrayContainer(
            IRandomizer randomizer,
            IFreeEnergyCalculator freeEnergyCalculator,
            int visibleNeuronCount, 
            int hiddenNeuronCount
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            //freeEnergyCalculator allowed to be null
            if (visibleNeuronCount <= 0 || hiddenNeuronCount <= 0)
            {
                throw new ArgumentException("visibleNeuronCount <= 0 || hiddenNeuronCount <= 0");
            }

            _freeEnergyCalculator = freeEnergyCalculator ?? new MockFreeEnergyCalculator();
            VisibleNeuronCount = visibleNeuronCount;
            HiddenNeuronCount = hiddenNeuronCount;

            _visibleNeuronCountWithBias = visibleNeuronCount + 1; //bias neuron
            _hiddenNeuronCountWithBias = hiddenNeuronCount + 1; //bias neuron

            //создаем массивы
            Input = new float[_visibleNeuronCountWithBias];
            Visible = new float[_visibleNeuronCountWithBias];
            Hidden0 = new float[_hiddenNeuronCountWithBias];
            Hidden1 = new float[_hiddenNeuronCountWithBias];
            Weights = new float[_visibleNeuronCountWithBias * _hiddenNeuronCountWithBias];
            _nabla = new float[_visibleNeuronCountWithBias * _hiddenNeuronCountWithBias];

            //инициализируем веса
            Weights.Fill(() => (randomizer.Next() * 0.02f - 0.01f));

            //заполняем bias входных объектов
            Input[VisibleNeuronCount ] = 1f;
            Visible[VisibleNeuronCount] = 1f;

            //заполняем bias выходных объектов
            Hidden0[HiddenNeuronCount] = 1f;
            Hidden1[HiddenNeuronCount] = 1f;
        }

        public void SetTrainItem(float[] input)
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }

            Array.Copy(input, Input, VisibleNeuronCount);
        }

        public void ClearNabla()
        {
            _nabla.Clear();
        }

        public void CalculateNabla()
        {
            Parallel.For(0, HiddenNeuronCount, hiddenIndex =>
                //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount - 1; hiddenIndex++)
            {
                //задаем отрицательную часть изменения весов
                for (var visibleIndex = 0; visibleIndex < VisibleNeuronCount; visibleIndex++)
                {
                    float error =
                        Input[visibleIndex] * Hidden0[hiddenIndex]
                        -
                        Visible[visibleIndex] * Hidden1[hiddenIndex];

                    _nabla[CalculateWeightIndex(hiddenIndex, visibleIndex)] += error;
                }
            }
                ); //Parallel.For
        }

        public void UpdateWeights(
            int batchSize,
            float learningRate)
        {
            for (var cc = 0; cc < _nabla.Length; cc++)
            {
                Weights[cc] += learningRate * _nabla[cc] / batchSize;
            }
        }

        public float GetError()
        {
            var sqdiff = 0.0f;
            for (var cc = 0; cc < VisibleNeuronCount; cc++)
            {
                var dln = (Visible[cc] - Input[cc]);
                sqdiff += dln * dln;
            }

            var result = (float) Math.Sqrt(sqdiff);

            return result;
        }

        public void Save(
            IArtifactContainer container)
        {
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }

            container.SaveSerialized(
                this.Weights,
                "weights.bin"
                );
        }

        public double CalculateFreeEnergy(
            IDataSet data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            return 
                _freeEnergyCalculator.CalculateFreeEnergy(
                    this.Weights,
                    data);
        }

        private int CalculateWeightIndex(
            int hiddenIndex,
            int visibleIndex
            )
        {
            return
                hiddenIndex * _visibleNeuronCountWithBias + visibleIndex;
        }
    }
}