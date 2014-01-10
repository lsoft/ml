using System;
using System.Runtime.InteropServices;
using MathNet.Numerics.Distributions;
using MyNN;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.ForwardPropagation.ForwardFactory;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNNConsoleApp.DerivativeChange;
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
                var seed = DateTime.Now.Millisecond;
                var fv = new FeatureVisualization(
                    new DefaultRandomizer(ref seed),
                    //SerializationHelper.LoadFromFile<MLP>("SDAE20140107000011 MLP2/mlp20140110053243.mynn"),
                    SerializationHelper.LoadFromFile<MLP>("SDAE20140107000011 MLP2/1st ae/20140107094406-perItemError=3,447127.mynn"),
                    new DropConnectOpenCLForwardPropagationFactory<OpenCLLayerInferenceNew16>(VectorizationSizeEnum.VectorizationMode16, 2500, 0.5f),
                    //new OpenCLForwardPropagationFactory(), 
                    1f);

                fv.Visualize(
                    new MNISTVisualizer(),
                    "features.bmp",
                    900,
                    true,
                    false);

                return;
                //*/

                /*
                Test0.Execute();
                return;
                //*/

                /*

                //Transpose0.Execute();
                //Transpose1.Execute();
                Transpose2.Execute();
                return;

                //*/

                /*
                
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
                mlp.AutoencoderCutTail();

                var sc = new SparseCalculator();
                var sparsePercent = sc.Calculate(
                    mlp,
                    vd);
                Console.WriteLine("Sparse percent: {0}", (sparsePercent * 100));
                Console.ReadLine();
                return;
                //*/



                MLP2TrainStackedAutoencoder.Train();
                //MLP2TuneSDAE.Tune();
                //MLP2TrainMLPBasedOnSDAE.Tune();
                //*/


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
