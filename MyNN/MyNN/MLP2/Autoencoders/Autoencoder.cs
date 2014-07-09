﻿using System;
using System.Linq;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.Autoencoders
{
    public class Autoencoder : IAutoencoder
    {
        private readonly IRandomizer _randomizer;
        private readonly IMLP _net;

        public Autoencoder(
            IRandomizer randomizer,
            IMLPFactory mlpFactory,
            string root,
            string folderName,
            params LayerInfo[] layerInfos)
        {
            //root, folderName  allowed to be null

            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (mlpFactory == null)
            {
                throw new ArgumentNullException("mlpFactory");
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

            _net = mlpFactory.CreateMLP(
                root,
                folderName,
                layerInfos.Select(j => j.ActivationFunction).ToArray(),
                layerInfos.Select(j => j.LayerSize).ToArray());

            ConsoleAmbientContext.Console.WriteLine("Network does not found. Created with conf: " + _net.GetLayerInformation());
        }

        public IMLP Train(
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
                throw new InvalidOperationException("!trainDataProvider.IsAutoencoderDataSet");
            }

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    _randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        _net,
                        config,
                        clProvider),
                    _net,
                    validation, 
                    config,
                    true);

                algo.Train(trainDataProvider);
            }

            return _net;
        }

    }
}
