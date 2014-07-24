using System;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BoltzmannMachines;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine
{
    public class RBM
    {
        private readonly IRandomizer _randomizer;
        private readonly IContainer _container;
        private readonly IImageReconstructor _imageReconstructor;
        
        private readonly IAlgorithm _algorithm;

        public RBM(
            IRandomizer randomizer,
            IContainer container,
            IAlgorithm algorithm,
            IImageReconstructor imageReconstructor
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }
            if (algorithm == null)
            {
                throw new ArgumentNullException("algorithm");
            }
            if (imageReconstructor == null)
            {
                throw new ArgumentNullException("imageReconstructor");
            }

            _randomizer = randomizer;
            _container = container;
            _algorithm = algorithm;
            _imageReconstructor = imageReconstructor;
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
                    _container.VisibleNeuronCount,
                   
                    _container.HiddenNeuronCount,
                    _algorithm.Name
                    )
                );

            _algorithm.PrepareTrain(batchSize);

            var epochNumber = 0;
            while (epochNumber < epocheCount)
            {
                var beforeEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "{0} Epoche {1:D5} {2}",
                        new string('-', 23),
                        epochNumber,
                        new string('-', 23))
                        );

                //скорость обучения на эту эпоху
                var learningRate = learningRateController.GetLearningRate(epochNumber);

                var epocheTrainData = trainData.CreateShuffledDataSet(_randomizer);

                _algorithm.PrepareBatch();

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoch learning rate = {0}",
                        learningRate));

                foreach (var batch in epocheTrainData.Split(batchSize))
                {
                    _container.ClearNabla();

                    var indexIntoBatch = 0;
                    foreach (var trainItem in batch)
                    {
                        //gibbs sampling

                        //заполняем видимое
                        _container.SetTrainItem(trainItem.Input);

                        _algorithm.CalculateSamples(
                            indexIntoBatch,
                            maxGibbsChainLength);

                        //считаем разницу и записываем ее в наблу
                        _container.CalculateNabla();

                        indexIntoBatch++;
                    }

                    _algorithm.BatchFinished();

                    _container.UpdateWeights(learningRate);
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

            var reconstructedImageCount = _imageReconstructor.GetReconstructedImageCount();

            var epocheError = 0f;

            for (var indexof = 0; indexof < validationData.Count; indexof++)
            {
                var d = validationData[indexof];

                //заполняем видимое
                _container.SetTrainItem(d.Input);

                var reconstructed = _algorithm.CalculateReconstructed();

                var itemError = _container.GetError();
                epocheError += itemError;

                if (indexof < reconstructedImageCount)
                {
                    _imageReconstructor.AddPair(
                        indexof,
                        reconstructed);
                }
            }

            var perItemError = epocheError/validationData.Count;

            _imageReconstructor.GetReconstructedBitmap().Save(
                string.Format(
                    "reconstruct{0}.bmp",
                    epocheNumber));

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Reconstruction per-item error: {0}",
                    perItemError));
        }
    }

}
