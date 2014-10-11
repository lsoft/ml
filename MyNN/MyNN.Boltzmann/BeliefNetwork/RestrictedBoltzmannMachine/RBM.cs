using System;
using System.Text;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.FeatureExtractor;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.Boltzmann.BoltzmannMachines;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine
{
    public class RBM : IRBM
    {
        private readonly IArtifactContainer _rbmContainer;
        private readonly IContainer _container;
        private readonly IImageReconstructor _imageReconstructor;
        private readonly IFeatureExtractor _featureExtractor;

        private readonly IAlgorithm _algorithm;

        public RBM(
            IArtifactContainer rbmContainer,
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
            _container = container;
            _algorithm = algorithm;
            _imageReconstructor = imageReconstructor ?? new MockImageReconstructor();
            _featureExtractor = featureExtractor ?? new MockFeatureExtractor();
        }

        public void Train(
            ITrainDataProvider trainDataProvider,
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

            #region формируем наборы для вычисления свободной энергии

            var trainFreeEnergySet = trainDataProvider.GetDataSet(0);
            var validationFreeEnergySet = validationData;

            #endregion

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

                //получаем данные для эпохи (уже shuffled)
                var epocheTrainData = trainDataProvider.GetDataSet(epochNumber);

                if (epocheTrainData.InputLength != _container.VisibleNeuronCount)
                {
                    throw new InvalidOperationException("Размер датасета не совпадает с количеством видимых нейронов RBM");
                }

                _algorithm.PrepareBatch();

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoch learning rate = {0}",
                        learningRate));

                var indexIntoEpoch = 0;
                foreach (var batch in epocheTrainData.Split(batchSize))
                {
                    ConsoleAmbientContext.Console.Write(
                        string.Format(
                            "Processed {0} out of {1}",
                            indexIntoEpoch * batchSize,
                            epocheTrainData.Count));
                    ConsoleAmbientContext.Console.ReturnCarriage();

                    _container.ClearNabla();

                    var indexIntoBatch = 0;
                    foreach (var trainItem in batch)
                    {
                        //заполняем видимое
                        _container.SetInput(trainItem.Input);

                        _algorithm.ExecuteGibbsSampling(
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

                    indexIntoEpoch++;
                }

                this.CalculateFreeEnergy(
                    _rbmContainer,
                    trainFreeEnergySet,
                    validationData
                    );

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

                if (accuracyController.IsLastEpochBetterThanPrevious())
                {
                    _container.Save(epochContainer);
                }

                epochNumber++;

                var afterEpoch = DateTime.Now;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Epoche takes {0}",
                        (afterEpoch - beforeEpoch)));
                ConsoleAmbientContext.Console.WriteLine(new string('-', 60));
            }


        }

        private void CalculateFreeEnergy(
            IArtifactContainer rbmContainer,
            IDataSet trainFreeEnergySet,
            IDataSet validationFreeEnergySet
            )
        {
            if (rbmContainer == null)
            {
                throw new ArgumentNullException("rbmContainer");
            }
            if (trainFreeEnergySet == null)
            {
                throw new ArgumentNullException("trainFreeEnergySet");
            }
            if (validationFreeEnergySet == null)
            {
                throw new ArgumentNullException("validationFreeEnergySet");
            }

            var feTrain = _container.CalculateFreeEnergy(trainFreeEnergySet);
            var feTrainPerItem = feTrain / trainFreeEnergySet.Count;
            Console.WriteLine(
                "TRAIN per-item free energy = {0}",
                feTrainPerItem);

            var feValidation = _container.CalculateFreeEnergy(validationFreeEnergySet);
            var feValidationPerItem = feValidation / validationFreeEnergySet.Count;
            Console.WriteLine(
                "VALIDATION per-item free energy = {0}",
                feValidationPerItem);

            var diffPerItem = Math.Abs(feTrainPerItem - feValidationPerItem);
            Console.WriteLine(
                "-------> Per-item diff free energy {0} <-------",
                diffPerItem);

            using (var writeStream = rbmContainer.GetWriteStreamForResource("_free_energy.csv"))
            {
                var s = string.Format(
                    "{0};{1};{2};{3};{4}\r\n",
                    feTrain,
                    feValidation,
                    feTrainPerItem,
                    feValidationPerItem,
                    diffPerItem);

                var bytes = Encoding.ASCII.GetBytes(s);

                writeStream.Write(bytes, 0, bytes.Length);
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
                _container.SetInput(d.Input);

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
