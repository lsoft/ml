﻿using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment6
    {
        public static void Execute()
        {
            var rndSeed = 1872390;
            var randomizer = 
                new DefaultRandomizer(++rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            const float p = 0.5f;

            const int sampleCount = 2500;

            var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));
            

            var mlpf = new MLPFactory(
                layerFactory
                );

            var mlp = mlpf.CreateMLP(
                folderName,
                new IFunction[]
                {
                    null,
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                },
                new int[]
                {
                    784,
                    1200,
                    1200,
                    2200,
                    10
                });


            //var mlp = SerializationHelper.LoadFromFile<MLP>(
            //    "_DropConnectMLP20131215225141/epoche 97/20131217064313-9851 out of 98,51%.mynn");

            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.02f, 0.998f),
                    1,
                    0.0f,
                    1000,
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
                        new DropConnectBitCPUBackpropagationEpocheTrainer<VectorizedCPULayerInferenceV2>(
                            randomizer,
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            config,
                            clProvider,
                            sampleCount,
                            p),
                        new FileSystemMLPContainer(".", serialization),
                        mlp,
                        validation,
                        config);

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainData));
            }

            Console.WriteLine("Experiment #6 finished");
            Console.ReadLine();
        }
    }
}