using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using MathNet.Numerics.Distributions;
using MyNN;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;

using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNNConsoleApp.ClassificationAutoencoder;
using MyNNConsoleApp.DerivativeChange;
using MyNNConsoleApp.DropConnectInference;
using MyNNConsoleApp.MLP2;
using MyNNConsoleApp.NLNCA;
using MyNNConsoleApp.Nvidia;
using MyNNConsoleApp.PingPong;
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
                //pabProfiler.Main2();
                //NvidiaForwardOptimizer.Optimize();
                //NvidiaBackpropOptimizer.Optimize();
                //NvidiaTransposeBackpropOptimizer.Optimize();
                //NvidiaBackpropSmallMLPOptimizer.Optimize();
                //NvidiaDoDfCalculatorOptimizer.Optimize();
                NvidiaDoDfCalculatorGeneration2Optimizer.Optimize();
                return;
                //*/

                /*
                var seed = DateTime.Now.Millisecond;
                var fv = new FeatureVisualization(
                    new DefaultRandomizer(ref seed),
                    SerializationHelper.LoadFromFile<MLP>("temp/20140117112110-perItemError=5,645099.mynn"),
                    //new CPUDropConnectForwardPropagationFactory<VectorizedCPULayerInferenceV2>(VectorizationSizeEnum.VectorizationMode16, 2500, 0.5f),
                    new CPUForwardPropagationFactory(), 
                    5,
                    5f);

                fv.Visualize(
                    new MNISTVisualizer(),
                    "features sdae.bmp",
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

                var mlp = SerializationHelper.LoadFromFile<MLP>("temp/20140117112110-perItemError=5,645099.mynn");
                mlp.AutoencoderCutTail();

                var sc = new SparseCalculator(mlp);
                
                float sparsePercent;
                float avgNonZeroCountPerItem;
                float avgValueOfNonZero;
                sc.Calculate(
                    vd,
                    out sparsePercent,
                    out avgNonZeroCountPerItem,
                    out avgValueOfNonZero);
                
                Console.WriteLine("Sparse: {0}%", (sparsePercent * 100));
                Console.WriteLine("Average non-zero count per-item: {0}", avgNonZeroCountPerItem);
                Console.WriteLine("Average value of non-zero values per-dataset: {0}", avgValueOfNonZero);
                Console.ReadLine();
                return;
                //*/

                /*
                //CATrainSCDAE.Train();
                //CATuneSCDAE.Tune();
                CATrainMLPBasedOnSCDAE.Tune();
                //*/

                /*
                //MLP2TrainStackedAutoencoder.Train();
                //MLP2TuneSDAE.Tune();
                MLP2TrainMLPBasedOnSDAE.Tune();
                //*/

                /*
                //NextAutoencoder.Execute();
                //NextMLP.Execute();
                //NextAutoencoder2.Execute();
                //NextMLP2.Execute();
                //*/

                
                //MLP2NLNCA.TrainNLNCA();
                //MLP2AutoencoderNLNCA.TrainAutoencoderNLNCA();
                //MLP2NLNCAAutoencoderTest.Test();
                MLP2TrainStackedNLNCAAutoencoder.Train();
                //*/
                
                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
