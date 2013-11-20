using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Validation;

namespace MyNN.Autoencoders
{
    public class Autoencoder
    {
        private readonly MultiLayerNeuralNetwork _net;

        public Autoencoder(
            ref int rndSeed,
            string root,
            string folderName,
            params LayerInfo[] layerInfos)
        {
            if (root == null)
            {
                throw new ArgumentNullException("root");
            }
            if (folderName == null)
            {
                throw new ArgumentNullException("folderName");
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

            _net = new MultiLayerNeuralNetwork(
                root,
                folderName,
                layerInfos.Select(j => j.ActivationFunction).ToArray(),
                ref rndSeed,
                layerInfos.Select(j => j.LayerSize).ToArray());

            Console.WriteLine("Network does not found. Created with conf: " + _net.DumpLayerInformation());
        }

        public MultiLayerNeuralNetwork Train(
            ref int rndSeed, 
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
            if (!validation.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("!validation.IsAuencoderDataSet");
            }

            using (var universe = new VNNCLProvider(_net))
            {
                //создаем объект просчитывающий сеть
                var computer =
                    new VOpenCLComputer(universe, true);

                _net.SetComputer(computer);

                var alg =
                    //new NaiveBackpropagationLearningAlgorithm(
                    //new OpenCLNaiveBackpropAlgorithm(
                    new VOpenCLBackpropAlgorithm(
                        _net,
                        config,
                        validation.Validate,
                        universe,
                        rndSeed);

                //обучение сети
                alg.Train(trainDataProvider.GetDeformationDataSet);
            }

            return _net;
        }

    }
}
