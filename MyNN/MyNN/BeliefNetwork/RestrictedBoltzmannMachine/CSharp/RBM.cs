using System;
using AForge;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.NegativeSampler;
using MyNN.BoltzmannMachines;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp
{
    public class RBM
    {
        private readonly IRandomizer _randomizer;
        private readonly ICalculator _calculator;
        private readonly IImageReconstructor _imageReconstructor;
        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;
        
        private readonly float[] _input;
        private readonly float[] _visible;
        private readonly float[] _hidden0;
        private readonly float[] _hidden1;
        private readonly float[] _weights;
        private readonly float[] _nabla;
        private readonly INegativeSampler _negativeSampler;

        public RBM(
            IRandomizer randomizer,
            ICalculator calculator,
            INegativeSamplerFactory negativeSamplerFactory,
            IImageReconstructor imageReconstructor,
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (calculator == null)
            {
                throw new ArgumentNullException("calculator");
            }
            if (imageReconstructor == null)
            {
                throw new ArgumentNullException("imageReconstructor");
            }
            if (visibleNeuronCount <= 0 || hiddenNeuronCount <= 0)
            {
                throw new ArgumentException("visibleNeuronCount <= 0 || hiddenNeuronCount <= 0");
            }

            _randomizer = randomizer;
            _calculator = calculator;
            _imageReconstructor = imageReconstructor;
            _visibleNeuronCount = visibleNeuronCount + 1; //bias neuron
            _hiddenNeuronCount = hiddenNeuronCount + 1; //bias neuron

            //создаем массивы
            _input = new float[_visibleNeuronCount];
            _visible = new float[_visibleNeuronCount];
            _hidden0 = new float[_hiddenNeuronCount];
            _hidden1 = new float[_hiddenNeuronCount];
            _weights = new float[_visibleNeuronCount * _hiddenNeuronCount];
            _nabla = new float[_visibleNeuronCount * _hiddenNeuronCount];

            //инициализируем веса
            _weights.Fill(() => (_randomizer.Next() * 0.02f - 0.01f));

            //заполняем bias входных объектов
            _input[_visibleNeuronCount - 1] = 1f;
            _visible[_visibleNeuronCount - 1] = 1f;

            //заполняем bias выходных объектов
            _hidden0[_hiddenNeuronCount - 1] = 1f;
            _hidden1[_hiddenNeuronCount - 1] = 1f;

            _negativeSampler = negativeSamplerFactory.CreateNegativeSampler(
                _calculator,
                _weights,
                _visible,
                _hidden0,
                _hidden1);
        }

        public void Train(
            IDataSet trainData,
            IDataSet validationData,
            ILearningRate learningRateController,
            int epocheCount,
            int batchSize,
            int maxGibbsChainLength
            )
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "RBM ({0}-{1}) training starts with algorightm {2}",
                    _visibleNeuronCount,
                    _hiddenNeuronCount,
                    _negativeSampler.Name
                    )
                );

            _negativeSampler.PrepareTrain(batchSize);

            var epochNumber = 0;
            while (epochNumber < epocheCount)
            {
                var beforeEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "{0} Epoche {1:D5} {2}",
                        new string('-', 22),
                        epochNumber,
                        new string('-', 22))
                        );

                //скорость обучения на эту эпоху
                var learningRate = learningRateController.GetLearningRate(epochNumber);

                var epocheTrainData = trainData.CreateShuffledDataSet(_randomizer);

                _negativeSampler.PrepareBatch();

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoch learning rate = {0}",
                        learningRate));

                foreach (var batch in epocheTrainData.Split(batchSize))
                {
                    this.ClearNabla();

                    var indexIntoBatch = 0;
                    foreach (var trainItem in batch)
                    {
                        //gibbs sampling

                        //заполняем видимое
                        Array.Copy(trainItem.Input, _input, _visibleNeuronCount - 1);

                        //sample hidden
                        _calculator.SampleHidden(
                            _weights,
                            _hidden0,
                            _input
                            );

                        _negativeSampler.CalculateNegativeSample(
                            indexIntoBatch,
                            maxGibbsChainLength);

                        //считаем разницу и записываем ее в наблу
                        this.CalculateNabla();

                        indexIntoBatch++;
                    }

                    _negativeSampler.BatchFinished();

                    this.UpdateWeights(learningRate);
                }

                this.CaculateError(
                    validationData,
                    epochNumber
                    );

                epochNumber++;

                var afterEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoche takes {0}",
                        (afterEpoch - beforeEpoch)));
                ConsoleAmbientContext.Console.WriteLine(new string('-', 60));
            }


        }

        private void CaculateError(
            IDataSet validationData,
            int epocheNumber
            )
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            const int reconstructedImageCount = 300;

            var epocheError = 0f;

            for (var indexof = 0; indexof < validationData.Count; indexof++)
            {
                var d = validationData[indexof];

                //заполняем видимое
                Array.Copy(d.Input, _visible, _visibleNeuronCount - 1);

                _calculator.CalculateHidden(
                    _weights,
                    _hidden0,
                    _visible);

                _calculator.CalculateVisible(
                    _weights, 
                    _visible,
                    _hidden0);

                var sqdiff = 0.0f;
                for (var cc = 0; cc < _visibleNeuronCount - 1; cc++)
                {
                    var dln = (_visible[cc] - d.Input[cc]);
                    sqdiff += dln * dln;
                }
                epocheError += (float)Math.Sqrt(sqdiff);

                if (indexof < reconstructedImageCount)
                {
                    _imageReconstructor.AddPair(
                        indexof,
                        _visible);
                }
                indexof++;
            }

            _imageReconstructor.GetReconstructedBitmap().Save(
                string.Format(
                    "reconstruct{0}.bmp",
                    epocheNumber));

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Epoche {0} error: {1}",
                    epocheNumber,
                    epocheError));
        }

        private void ClearNabla()
        {
            _nabla.Clear();
        }

        private void UpdateWeights(float learningRate)
        {
            for (var cc = 0; cc < _nabla.Length; cc++)
            {
                _weights[cc] += learningRate * _nabla[cc];
            }
        }

        private void CalculateNabla()
        {
            Parallel.For(0, _hiddenNeuronCount - 1, hiddenIndex => 
            //for (var hiddenIndex = 0; hiddenIndex < _hiddenNeuronCount - 1; hiddenIndex++)
            {
                //задаем отрицательную часть изменения весов
                for (var visibleIndex = 0; visibleIndex < _visibleNeuronCount - 1; visibleIndex++)
                {
                    float error =
                        _input[visibleIndex] * _hidden0[hiddenIndex]
                        - 
                        _visible[visibleIndex] * _hidden1[hiddenIndex];

                    _nabla[CalculateWeightIndex(hiddenIndex, visibleIndex)] += error;
                }
            }
            ); //Parallel.For
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
