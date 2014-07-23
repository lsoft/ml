using System;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;

namespace MyNNConsoleApp.PingPong
{
    public class NextAutoencoder2
    {
        public static void Execute()
        {
            var rndSeed = 885341;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //100
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            //через обученную сеть генерируем данные для следующей эпохи
            DataSet trainNext;
            DataSet validationNext;
            var mlpPath1 = "PingPong/Experiment0/Step0/20140115132416-9876 out of 98,76%.mynn";
            NextDataSet.NextDataSets(mlpPath1, trainData, validationData, out trainNext, out validationNext);

            DataSet trainNext2;
            DataSet validationNext2;
            var mlpPath2 = "PingPong/Experiment0/Step1/20140115181556-9886 out of 98,86%.mynn";
            NextDataSet.NextDataSets(mlpPath2, trainNext, validationNext, out trainNext2, out validationNext2);

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));
            

            var mlpf = new MLPFactory(
                layerFactory
                );

            //обучаем автоенкодер
            var a = new Autoencoder(
                new FileSystemMLPContainer(".", serialization), 
                randomizer,
                mlpf,
                DateTime.Now.Ticks.ToString(),
                new LayerInfo[]
                {
                    new LayerInfo(
                        trainNext2[0].Input.Length,
                        new RLUFunction()), 
                    new LayerInfo(
                        2200,
                        new RLUFunction()),
                    new LayerInfo(
                        trainNext2[0].Input.Length,
                        new RLUFunction()), 
                });

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.001f, 0.99f),
                1,
                0.0f,
                50,
                0.0001f,
                -1.0f);

            var noiser = new SetOfNoisers(
                randomizer,
                new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.25f)),
                new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.25f)),
                new Pair<float, INoiser>(0.34f, new GaussNoiser(0.20f, false))
                );

            var validation = new AutoencoderValidation(
                
                new HalfSquaredEuclidianDistance(),
                validationNext2.ConvertToAutoencoder(),
                300,
                100);

            a.Train(
                config,
                new NoiseDataProvider(trainNext2.ConvertToAutoencoder(), noiser),
                validation);
        }
    }
}
