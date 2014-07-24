using System;
using AForge;
using MathNet.Numerics.Distributions;
using MyNN.BoltzmannMachines;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp
{
    public class LNRELURBM
    {
        private readonly Normal _gaussRandom = new Normal(0, 1);

        private readonly IRandomizer _randomizer;
        private readonly IImageReconstructor _imageReconstructor;
        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;
        
        private readonly float[] _input;
        private readonly float[] _visible;
        private readonly float[] _hidden0;
        private readonly float[] _hidden1;
        private readonly float[] _weights;
        private readonly float[] _nabla;

        public LNRELURBM(
            IRandomizer randomizer,
            IImageReconstructor imageReconstructor,
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
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

            var epochNumber = 0;
            while (epochNumber < epocheCount)
            {
                //скорость обучения на эту эпоху
                var learningRate = learningRateController.GetLearningRate(epochNumber);


                var epocheTrainData = trainData.CreateShuffledDataSet(_randomizer);

                foreach (var batch in epocheTrainData.Split(batchSize))
                {
                    this.ClearNabla();

                    foreach (var trainItem in batch)
                    {
                        //gibbs sampling

                        //заполняем видимое
                        Array.Copy(trainItem.Input, _input, _visibleNeuronCount - 1);

                        //sample hidden
                        this.SampleHidden(
                            _hidden0,
                            _input
                            );

                        for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
                        {
                            var ifFirst = cdi == 0;
                            var ifLast = cdi == (maxGibbsChainLength - 1);

                            //compute visible
                            this.ComputeVisible(
                                _visible,
                                ifFirst ? _hidden0 : _hidden1
                                );

                            if (ifLast)
                            {
                                //compute hidden
                                this.ComputeHidden(
                                    _hidden1,
                                    _visible);
                            }
                            else
                            {
                                //sample hidden
                                this.SampleHidden(
                                    _hidden1,
                                    _visible
                                    );
                            }

                        }

                        //считаем разницу и записываем ее в наблу
                        this.CalculateNabla();
                    }

                    this.UpdateWeights(learningRate);
                }

                this.CalculateError(
                    validationData,
                    epochNumber
                    );

                epochNumber++;
            }


        }

        private void CalculateError(
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

                this.ComputeHidden(
                    _hidden0,
                    _visible);
                //this.SampleHidden(
                //    _hidden0,
                //    _visible);

                this.ComputeVisible(
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

        private void ComputeVisible(
            float[] targetVisible,
            float[] fromHidden
            )
        {
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
                        _weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }

                targetVisible[visibleIndex] = sum;
            }
            );//Parallel.For
        }

        private void SampleVisible(
            float[] targetVisible,
            float[] fromHidden
            )
        {
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
                        _weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]
                        * fromHidden[hiddenIndex];
                }

                targetVisible[visibleIndex] = sum;
            }
            );//Parallel.For
        }


        private void ComputeHidden(
            float[] targetHidden,
            float[] fromVisible
            )
        {
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
                    sum += _weights[CalculateWeightIndex(hiddenIndex, visibleIndex)] * fromVisible[visibleIndex];
                }

                targetHidden[hiddenIndex] = sum;
            }
            );//Parallel.For
        }

        private void SampleHidden(
            float[] targetHidden,
            float[] fromVisible
            )
        {
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
                    sum += _weights[CalculateWeightIndex(hiddenIndex, visibleIndex)]*fromVisible[visibleIndex];
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

        public float ComputeSigmoid(float x)
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
