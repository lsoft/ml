using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN
{
    public class DeepBeliefNetwork : IDisposable
    {
        public const string RbmFolderName = "rbm_layer";

        private readonly IRandomizer _randomizer;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly int _reconstructedImageCount;
        private readonly INegativeSamplerProvider _samplerProvider;
        private readonly int[] _layerSizes;
        private readonly CLProvider _clProvider;

        public RestrictedBoltzmannMachine[] RBMList
        {
            get;
            private set;
        }

        public DeepBeliefNetwork(
            IRandomizer randomizer,
            int imageWidth,
            int imageHeight,
            int reconstructedImageCount,
            INegativeSamplerProvider samplerProvider,
            params int[] layerSizes)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (layerSizes == null || layerSizes.Length < 2)
            {
                throw new ArgumentException("layerSizes");
            }
            if (imageHeight*imageWidth != layerSizes[0])
            {
                throw new ArgumentException("imageHeight*imageWidth != layerSizes[0]");
            }

            _randomizer = randomizer;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _reconstructedImageCount = reconstructedImageCount;
            _samplerProvider = samplerProvider;
            _layerSizes = layerSizes;

            RBMList = new RestrictedBoltzmannMachine[layerSizes.Length];
            _clProvider = new CLProvider();

            for (var cc = 0; cc < _layerSizes.Length - 1; cc++)
            {
                var rbm = new RestrictedBoltzmannMachine(
                    _randomizer,
                    _clProvider,
                    _layerSizes[cc], 
                    _layerSizes[cc + 1]);

                var sampler = this._samplerProvider.GetNegativeSampler(rbm);

                rbm.SetNegativeSampler(sampler);

                RBMList[cc] = rbm;
            }

        }

        public void Train(
            DataSet trainData,
            DataSet validationData,
            int batchSize,
            ILearningRate learningRate,
            float errorThreshold,
            int epochThreshold,
            string artifactFolderRoot, 
            int maxGibbsChainLength)
        {
            Directory.CreateDirectory(artifactFolderRoot);

            DumpDBNInfo(artifactFolderRoot);

            //RBMList[0].LoadWeights("C:/Users/ls.OFFICE/Desktop/float-deep-autoencoder/dbn20130301093130/rbm_layer0/epoche 67/weights.txt");
            //RBMList[1].LoadWeights("C:/Users/ls.OFFICE/Desktop/float-deep-autoencoder/dbn20130301093130/rbm_layer1/epoche 432/weights.txt");
            //RBMList[2].LoadWeights("C:/Users/ls.OFFICE/Desktop/float-deep-autoencoder/dbn20130301093130/rbm_layer2/epoche 48/weights.txt");

            var layerTrainData =
                trainData;
                //RBMList[1].SampleHidden(RBMList[0].SampleHidden(trainData));

            var layerValidationData =
                validationData;
                //RBMList[1].SampleHidden(RBMList[0].SampleHidden(validationData));

            for (var layerIndex =
                            0;
                            //2;
                            layerIndex < _layerSizes.Length - 1; layerIndex++)
            {
                var rbm = RBMList[layerIndex];

                rbm.Train(
                    layerTrainData,
                    layerValidationData,
                    batchSize,
                    learningRate,
                    errorThreshold,
                    epochThreshold,
                    Path.Combine(artifactFolderRoot, DeepBeliefNetwork.RbmFolderName + layerIndex),
                    new DBNFeatureExtractor(layerIndex, _layerSizes[layerIndex + 1], _imageWidth, _imageHeight),
                    new DBNImageReconstructor(this, layerIndex, validationData, _reconstructedImageCount, _imageWidth, _imageHeight),
                    _reconstructedImageCount,
                    maxGibbsChainLength);

                //File.WriteAllText("é.txt", string.Join(
                //    "\r\n",
                //    trainData.ConvertAll(j => string.Join("", j.Input.ToList().ConvertAll(k => k.ToString())))));

                layerTrainData = rbm.ExecuteSampleHidden(layerTrainData);
                layerValidationData = rbm.ExecuteSampleHidden(layerValidationData);

                //File.WriteAllText("û.txt", string.Join(
                //    "\r\n",
                //    trainData.ConvertAll(j => string.Join("", j.Input.ToList().ConvertAll(k => k.ToString())))));
            }
        }

        private void DumpDBNInfo(string artifactFolderRoot)
        {
            var path = Path.Combine(artifactFolderRoot, "dbn.info");

            File.WriteAllText(
                path,
                "Train algorithm: " + this._samplerProvider.Name);

            File.AppendAllText(
                path,
                "\r\n");

            File.AppendAllText(
                path,
                "Layers sizes: " + string.Join("-", _layerSizes.ToList().ConvertAll(k => k.ToString()).ToArray()));

            File.AppendAllText(
                path,
                "\r\n");
        }

        public void Dispose()
        {
            _clProvider.Dispose();
        }
    }
}
