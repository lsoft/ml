using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Metadata.W3cXsd2001;
using System.Text;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2TrainMLPBasedOnSDAE
    {
        public static void Tune()
        {
            var rndSeed = 74787;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //1000
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            var mlp = new SerializationHelper().LoadFromFile<MLP>(
                "MLP20140201224821/epoche 49/20140202141949-(1,95896).mynn");
                //"SDAE20140113100839 MLP2/MLP20140113100840/epoche 44/20140113121811-perItemError=2,562213.mynn");

            mlp.AutoencoderCutTail();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                //new DRLUFunction(), 
                10,
                false);

            Console.WriteLine("Network configuration: " + mlp.GetLayerInformation());


            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.02f, 0.98f),
                    1,
                    0.0f,
                    50,
                    0.0001f,
                    -1.0f);

                var validation = new ClassificationValidation(
                    
                    new HalfSquaredEuclidianDistance(), 
                    validationData,
                    300,
                    100);

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        new CPUBackpropagationEpocheTrainer(
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            config,
                            clProvider),
                        new FileSystemMLPContainer(".", serialization),
                        mlp,
                        validation,
                        config);

               //var noiser = new ZeroMaskingNoiser(randomizer, 0.1f);

                //var noiser = new SetOfNoisers(
                //    randomizer,
                //    new Pair<float, INoiser>(0.25f, new ZeroMaskingNoiser(randomizer, 0.10f)),
                //    new Pair<float, INoiser>(0.25f, new SaltAndPepperNoiser(randomizer, 0.10f)),
                //    new Pair<float, INoiser>(0.25f, new GaussNoiser(0.05f, false)),
                //    new Pair<float, INoiser>(0.25f, new MultiplierNoiser(randomizer, 0.2f))
                //    );

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainData));
            }


        }
    }
}
