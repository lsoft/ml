using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using MyNN.BeliefNetwork.FeatureExtractor;
using MyNN.BeliefNetwork.ImageReconstructor;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BoltzmannMachines;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine
{
    public class RBM
    {
        private readonly IArtifactContainer _rbmContainer;
        private readonly IRandomizer _randomizer;
        private readonly IContainer _container;
        private readonly IImageReconstructor _imageReconstructor;
        private readonly IFeatureExtractor _featureExtractor;

        private readonly IAlgorithm _algorithm;

        public RBM(
            IArtifactContainer rbmContainer,
            IRandomizer randomizer,
            IContainer container,
            IAlgorithm algorithm,
            IImageReconstructor imageReconstructor,
            IFeatureExtractor featureExtractor
            )
        {
            if (rbmContainer == null)
            {
                throw new ArgumentNullException("rbmContainer");
            }
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
            //imageReconstructor allowed to be null
            //featureExtractor allowed to be null

            _rbmContainer = rbmContainer;
            _randomizer = randomizer;
            _container = container;
            _algorithm = algorithm;
            _imageReconstructor = imageReconstructor ?? new MockImageReconstructor();
            _featureExtractor = featureExtractor ?? new MockFeatureExtractor();
        }

        public void Train(
            Func<IDataSet> trainDataProvider,
            IDataSet validationData,
            ILearningRate learningRateController,
            IAccuracyController accuracyController,
            int batchSize,
            int maxGibbsChainLength
            )
        {
            if (trainDataProvider == null)
            {
                throw new ArgumentNullException("trainDataProvider");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }
            if (accuracyController == null)
            {
                throw new ArgumentNullException("accuracyController");
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
            while (!accuracyController.IsNeedToStop())
            {
                var beforeEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "{0} Epoche {1:D5} {2}",
                        new string('-', 23),
                        epochNumber,
                        new string('-', 23))
                        );

                var epochContainer = _rbmContainer.GetChildContainer(
                    string.Format(
                        "epoch {0}",
                        epochNumber));

                //скорость обучения на эту эпоху
                var learningRate = learningRateController.GetLearningRate(epochNumber);

                var freshTrainData = trainDataProvider();
                var epocheTrainData = freshTrainData.CreateShuffledDataSet(_randomizer);

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

                    _container.UpdateWeights(
                        batchSize,
                        learningRate);
                }

                var error = this.CaculateError(
                    epochContainer,
                    validationData
                    );

                this.SaveFeatures(
                    epochContainer
                    );

                accuracyController.AddError(
                    epochNumber,
                    error);

                epochNumber++;

                var afterEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoche takes {0}",
                        (afterEpoch - beforeEpoch)));
                ConsoleAmbientContext.Console.WriteLine(new string('-', 60));
            }


        }

        private void SaveFeatures(
            IArtifactContainer epochContainer
            )
        {
            if (epochContainer == null)
            {
                throw new ArgumentNullException("epochContainer");
            }
            var features = _algorithm.GetFeatures();

            foreach (var feature in features)
            {
                _featureExtractor.AddFeature(feature);
            }

            using (var writeStream = epochContainer.GetWriteStreamForResource("feature.bmp"))
            {
                var bitmap = _featureExtractor.GetFeatureBitmap();

                bitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
            }
        }

        private float CaculateError(
            IArtifactContainer epochContainer,
            IDataSet validationData
            )
        {
            if (epochContainer == null)
            {
                throw new ArgumentNullException("epochContainer");
            }
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

            using (var writeStream = epochContainer.GetWriteStreamForResource("reconstruct.bmp"))
            {
                var bitmap = _imageReconstructor.GetReconstructedBitmap();

                bitmap.Save(writeStream, System.Drawing.Imaging.ImageFormat.Bmp);
            }

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Reconstruction per-item error: {0}",
                    perItemError));

            return
                perItemError;
        }
    }

}
