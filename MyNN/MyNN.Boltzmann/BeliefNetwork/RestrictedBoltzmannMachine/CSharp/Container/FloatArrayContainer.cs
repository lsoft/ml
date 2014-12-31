using System;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container
{
    public class FloatArrayContainer : IContainer
    {
        private readonly IFreeEnergyCalculator _freeEnergyCalculator;

        private readonly float[] _nablaWeights;
        private readonly float[] _nablaVisibleBiases;
        private readonly float[] _nablaHiddenBiases;

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

        public float[] VisibleBiases
        {
            get;
            private set;
        }

        public float[] HiddenBiases
        {
            get;
            private set;
        }
        
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

            //создаем массивы
            Input = new float[VisibleNeuronCount];
            Visible = new float[VisibleNeuronCount];
            Hidden0 = new float[HiddenNeuronCount];
            Hidden1 = new float[HiddenNeuronCount];

            Weights = new float[VisibleNeuronCount * HiddenNeuronCount];
            VisibleBiases = new float[VisibleNeuronCount];
            HiddenBiases = new float[HiddenNeuronCount];

            _nablaWeights = new float[VisibleNeuronCount * HiddenNeuronCount];
            _nablaVisibleBiases = new float[VisibleNeuronCount];
            _nablaHiddenBiases = new float[HiddenNeuronCount];

            //инициализируем веса
            Weights.Fill(() => (randomizer.Next() * 0.02f - 0.01f));
            VisibleBiases.Fill(() => (randomizer.Next() * 0.02f - 0.01f));
            HiddenBiases.Fill(() => (randomizer.Next() * 0.02f - 0.01f));
        }

        public void SetInput(float[] input)
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }

            Array.Copy(input, Input, VisibleNeuronCount);
        }

        public void SetHidden(float[] hidden)
        {
            if (hidden == null)
            {
                throw new ArgumentNullException("hidden");
            }

            Array.Copy(hidden, Hidden0, HiddenNeuronCount);
        }

        public void ClearNabla()
        {
            _nablaWeights.Clear();
            _nablaVisibleBiases.Clear();
            _nablaHiddenBiases.Clear();
        }

        public void CalculateNabla()
        {
            Parallel.For(0, HiddenNeuronCount, hiddenIndex =>
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount; hiddenIndex++)
            {
                //задаем отрицательную часть изменения весов
                for (var visibleIndex = 0; visibleIndex < VisibleNeuronCount; visibleIndex++)
                {
                    float error =
                        Input[visibleIndex] * Hidden0[hiddenIndex]
                        -
                        Visible[visibleIndex] * Hidden1[hiddenIndex];

                    _nablaWeights[CalculateWeightIndex(hiddenIndex, visibleIndex)] += error;
                }
            }
            ); //Parallel.For

            for (var visibleIndex = 0; visibleIndex < VisibleNeuronCount; visibleIndex++)
            {
                var error = Input[visibleIndex] - Visible[visibleIndex];

                _nablaVisibleBiases[visibleIndex] += error;
            }

            for (var hiddenIndex = 0; hiddenIndex < HiddenNeuronCount; hiddenIndex++)
            {
                var error = Hidden0[hiddenIndex] - Hidden1[hiddenIndex];

                _nablaHiddenBiases[hiddenIndex] += error;
            }
        }

        public void UpdateWeights(
            int batchSize,
            float learningRate)
        {
            for (var cc = 0; cc < _nablaWeights.Length; cc++)
            {
                Weights[cc] += learningRate * _nablaWeights[cc] / batchSize;
            }
            for (var cc = 0; cc < _nablaVisibleBiases.Length; cc++)
            {
                VisibleBiases[cc] += learningRate * _nablaVisibleBiases[cc] / batchSize;
            }
            for (var cc = 0; cc < _nablaHiddenBiases.Length; cc++)
            {
                HiddenBiases[cc] += learningRate * _nablaHiddenBiases[cc] / batchSize;
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

            var saveableContainer = new SaveableContainer(
                this.Weights,
                this.VisibleBiases,
                this.HiddenBiases
                );

            container.SaveSerialized(
                saveableContainer,
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

            //ConsoleAmbientContext.Console.WriteWarning(
            //    "Avg weights {0}",
            //    this.Weights.Average()
            //    );
            //ConsoleAmbientContext.Console.WriteWarning(
            //    "Avg visible biases {0}",
            //    this.VisibleBiases.Average()
            //    );
            //ConsoleAmbientContext.Console.WriteWarning(
            //    "Avg hidden biases {0}",
            //    this.HiddenBiases.Average()
            //    );

            return 
                _freeEnergyCalculator.CalculateFreeEnergy(
                    this.Weights,
                    this.VisibleBiases,
                    this.HiddenBiases,
                    data);
        }

        private int CalculateWeightIndex(
            int hiddenIndex,
            int visibleIndex
            )
        {
            return
                hiddenIndex * VisibleNeuronCount + visibleIndex;
        }
    }
}