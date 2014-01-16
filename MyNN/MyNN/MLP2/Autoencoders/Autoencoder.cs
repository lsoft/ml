﻿using System;
using System.Linq;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;

namespace MyNN.MLP2.Autoencoders
{
    public class Autoencoder
    {
        private readonly IRandomizer _randomizer;
        private readonly MLP _net;

        public Autoencoder(
            IRandomizer randomizer,
            string root,
            string folderName,
            params LayerInfo[] layerInfos)
        {
            //root, folderName  allowed to be null

            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (layerInfos == null)
            {
                throw new ArgumentNullException("layerInfos");
            }
            if (layerInfos.Length < 3)
            {
                throw new ArgumentException("layerInfos");
            }
            if (layerInfos.First().LayerSize != layerInfos.Last().LayerSize)
            {
                throw new ArgumentException("layerInfos sizes");
            }

            _randomizer = randomizer;

            _net = new MLP(
                randomizer,
                root,
                folderName,
                layerInfos.Select(j => j.ActivationFunction).ToArray(),
                layerInfos.Select(j => j.LayerSize).ToArray());

            ConsoleAmbientContext.Console.WriteLine("Network does not found. Created with conf: " + _net.DumpLayerInformation());
        }

        public MLP Train(
            ILearningAlgorithmConfig config, 
            ITrainDataProvider trainDataProvider,
            IValidation validation)
        {
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (validation == null)
            {
                throw new ArgumentNullException("validation");
            }
            if (trainDataProvider == null)
            {
                throw new ArgumentNullException("trainDataProvider");
            }
            if (!trainDataProvider.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("!trainDataProvider.IsAuencoderDataSet");
            }

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    _randomizer,
                    (processedMLP, processedConfig) => new OpenCLBackpropagationAlgorithm(
                        VectorizationSizeEnum.VectorizationMode16,
                        processedMLP,
                        processedConfig,
                        clProvider),
                    _net,
                    validation, 
                    config,
                    true);

                algo.Train(trainDataProvider.GetDeformationDataSet);
            }

            return _net;
        }

    }
}
