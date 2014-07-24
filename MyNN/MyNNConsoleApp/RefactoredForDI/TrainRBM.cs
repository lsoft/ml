using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.Container;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainRBM
    {
        public static void DoTrainLNRELU()
        {
            var randomizer = new DefaultRandomizer(123);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000//int.MaxValue
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
                );
            validationData.GNormalize();

            const int visibleNeuronCount = 784;
            const int hiddenNeuronCount = 500;

            var calculator = new LNRELUCalculator(visibleNeuronCount, hiddenNeuronCount);

            var container = new FloatArrayContainer(randomizer, visibleNeuronCount, hiddenNeuronCount);

            var algorithm = new CD(
                calculator,
                container);

            var reconstructor = new IsolatedImageReconstructor(
                validationData, 
                300, 
                28, 
                28);

            var extractor = new IsolatedFeatureExtractor(
                hiddenNeuronCount, 
                28, 
                28);

            var rbm = new RBM(
                randomizer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            rbm.Train(
                () => trainData,
                validationData,
                new ConstLearningRate(0.0001f),
                20,
                10,
                1);
        }

        public static void DoTrainBB()
        {
            var randomizer = new DefaultRandomizer(123);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000
                //int.MaxValue
                );

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
                );
            validationData = validationData.Binarize(randomizer);

            const int visibleNeuronCount = 784;
            const int hiddenNeuronCount = 500;

            var calculator = new BBCalculator(randomizer, visibleNeuronCount, hiddenNeuronCount);

            var container = new FloatArrayContainer(randomizer, visibleNeuronCount, hiddenNeuronCount);

            var algorithm = new CD(
                calculator,
                container);

            var reconstructor = new IsolatedImageReconstructor(
                validationData, 
                300, 
                28, 
                28);

            var extractor = new IsolatedFeatureExtractor(
                hiddenNeuronCount, 
                28, 
                28);

            var rbm = new RBM(
                randomizer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            rbm.Train(
                () =>
                {
                    var binarized = trainData.Binarize(randomizer);
                    return binarized;
                },
                validationData,
                new LinearLearningRate(0.01f, 0.99f), 
                20,
                10,
                1);
        }
    }
}
