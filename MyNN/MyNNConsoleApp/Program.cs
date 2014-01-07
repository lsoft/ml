﻿using System;
using System.Runtime.InteropServices;
using MathNet.Numerics.Distributions;
using MyNN;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNNConsoleApp.DropConnectInference;
using MyNNConsoleApp.MLP2;
using MyNNConsoleApp.TransposeExperiments;
using OpenCvSharp;

namespace MyNNConsoleApp
{
    class Program
    {
        private static void Main(string[] args)
        {
            const float DeltaX = 0.01f;
            const float DerivativeEpsilon = 0.001f;

            var f = new SigmoidFunction(1f);

            var x = 0f;

            var center = f.Compute(x);
            var left = f.Compute(x - DeltaX);
            var right = f.Compute(x + DeltaX);

            var cDerivative = (right - left) / (2f * DeltaX);
            var fDerivative = f.ComputeFirstDerivative(x);//center);

            var diff = Math.Abs(cDerivative - fDerivative);

            if (diff >= DerivativeEpsilon)
            {
                throw new Exception();
            }


            Console.WriteLine("45456546546546546546546546546654654645");





            using (new CombinedConsole("console.log"))
            {
                /*

                //Transpose0.Execute();
                //Transpose1.Execute();
                Transpose2.Execute();
                return;

                //*/

                
                
                //Experiment4.Execute();
                //Experiment6.Execute();
                TrainStackedAutoencoder.Train();
                //TuneSDAE.Tune();
                //TrainMLPBasedOnSDAE.Train();
                return;
                
                //*/

                /*

                var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");
                int rndSeed = 5482;
                var mlp = new MLP(
                    new DefaultRandomizer(ref rndSeed), 
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new RLUFunction(), 
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        4,
                        3,
                        2
                    });

                var scheme = mlp.GetVisualScheme();
                scheme.Save("_scheme.bmp");
                return;


                //*/

                /*
                var vd = MNISTDataProvider.GetDataSet(
                    "_MNIST_DATABASE/mnist/testset/",
                    int.MaxValue
                    );
                vd.Normalize();

                var mlp = SerializationHelper.LoadFromFile<MLP>("20131223081806-perItemError=3,892169.mynn");
                mlp.AutoencoderCut();

                var sc = new SparseCalculator();
                var sparsePercent = sc.Calculate(
                    mlp,
                    vd);
                Console.WriteLine("Sparse percent: {0}", (sparsePercent * 100));
                Console.ReadLine();
                return;
                //*/

                //MLP2TrainStackedAutoencoder.Train();
                //MLP2TuneSDAE.Tune();
                MLP2TrainMLPBasedOnSDAE.Tune();

                //MLP2NLNCA.TrainNLNCA();

                //MLP2AutoencoderNLNCA.TrainAutoencoderNLNCA();

                //MLP2NLNCAAutoencoderTest.Test();

                //MLP2TrainStackedNLNCAAutoencoder.Train();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
