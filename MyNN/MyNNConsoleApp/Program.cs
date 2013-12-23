using System;
using System.Runtime.InteropServices;
using MathNet.Numerics.Distributions;
using MyNN;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
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
            using (new CombinedConsole("console.log"))
            {
                /*

                //Transpose0.Execute();
                //Transpose1.Execute();
                Transpose2.Execute();
                return;

                //*/

                
                
                Experiment4.Execute();
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
