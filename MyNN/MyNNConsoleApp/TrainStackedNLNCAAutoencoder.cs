using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra.Complex.Factorization;
using MyNN;
using MyNN.Autoencoders;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train.Metrics;
using MyNN.NeuralNet.Train.Validation;

namespace MyNNConsoleApp
{
    public class TrainStackedNLNCAAutoencoder
    {
        public void Train(
            ref int rndSeed)
        {
            var root = "SDAE" + DateTime.Now.ToString("yyyyMMddHHmmss");

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true);

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                100,
                true);

            var noiser =
                new ZeroMaskingNoiser(ref rndSeed, 0.25f);

            var firstLayerSize = trainData[0].Input.Length;

            const float lambda = 0.1f;
            const float partTakeOfAccount = 0.5f;

            var first = true;
            var sa = new StackedNLNCAAutoencoder(
                (DataSet td) =>
                {
                    ITrainDataProvider result = 
                        first
                            ? (ITrainDataProvider)
                                    //new NoDeformationTrainDataProvider(td)
                                    new NoiseDataProvider(
                                        td,
                                        new GaussNoiser(0.025f, false))

                            : (ITrainDataProvider)new NoiseDataProvider(
                                td,
                                noiser);

                    first = false;

                    return result;
                },
                (DataSet vd) =>
                {
                    return
                        new AutoencoderValidation(
                            vd.ConvertToAutoencoder(),
                            300,
                            100);
                },
                (int depthIndex) =>
                {
                    var lr = 0.002f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        100,
                        0.0f,
                        50,
                        0f,
                        -0.001f,
                        new HalfSquaredEuclidianDistance());

                    return conf;
                },
                lambda,
                partTakeOfAccount,
                new LayerInfo(firstLayerSize, new RLUFunction()),
                new LayerInfo(600, new RLUFunction()),
                new LayerInfo(600, new RLUFunction()),
                new LayerInfo(2200, new RLUFunction())
                );

            if (!Directory.Exists(root))
            {
                Directory.CreateDirectory(root);
            }

            var combinedNet = sa.Train(
                ref rndSeed,
                root,
                trainData,
                validationData
                );

            //combinedNet.SetComputer(new DefaultComputer(combinedNet));


            //var x = MNISTDataProvider.GetDataSet(
            //    "mnist/testset/",
            //    10);
            //x.Normalize();

            //var o = combinedNet.ComputeOutput(x[0].Input);

            //var mv = new MNISTVisualizer();
            //mv.SaveAsPairList("_.bmp", new List<Pair<float[], float[]>>
            //{
            //    new Pair<float[], float[]>(x[0].Input, o)
            //});

            
            int g = 0;
        }
    }
}
