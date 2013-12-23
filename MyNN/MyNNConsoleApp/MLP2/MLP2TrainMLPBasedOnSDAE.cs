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
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2TrainMLPBasedOnSDAE
    {
        public static void Tune()
        {
            var rndSeed = 74783;
            var randomizer = new DefaultRandomizer(ref rndSeed);

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

            var mlp = SerializationHelper.LoadFromFile<MLP>(
                //"SDAE20131209230003 MLP2/mlp20131210181303.mynn");
                //"MLP20131218124915/epoche 42/20131219100700-perItemError=3,6219.mynn");
                "MLP20131221192758/epoche 56/20131223081806-perItemError=3,892169.mynn");
                

            mlp.AutoencoderCut();

            mlp.AddLayer(
                new SigmoidFunction(1f),
                //new IRLUFunction(), 
                10,
                false);

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


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
                    serialization,
                    new HalfSquaredEuclidianDistance(), 
                    validationData,
                    300,
                    100);

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new OpenCLBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
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
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
                    //new NoiseDataProvider(trainData, noiser).GetDeformationDataSet);
            }


        }
    }
}
