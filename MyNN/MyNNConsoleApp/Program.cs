using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using MathNet.Numerics.Distributions;
using MyNN;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.BoltzmannMachines.DBNInfo;
using MyNN.Data;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;

using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using MyNNConsoleApp.RefactoredForDI;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCvSharp;

namespace MyNNConsoleApp
{
    class Program
    {
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                //TrainMLP.DoTrain();
                //TrainAutoencoder.DoTrain();
                //TrainNLNCAMLP.DoTrain();
                //TrainNLNCAAutoencoder.DoTrain();

                //TrainRBM.DoTrainBB();
                //TrainRBM.DoTrainLNRELU();

                //TrainDBN.DoTrainBB();
                //TrainDBN.DoTrainLNRELU();
                //TrainDBN.DoTrainMLPOnDBN();
                //TrainDBN.DoTrainAutoencoder();
                //TrainDBN.DoTrainMLPOnAE();

                //TrainSDAE.DoTrain();

                //TrainMLPWithNoNoise.DoTrain();

                /*
                var randomizer = new DefaultRandomizer(123);

                //var noiser = new SequenceNoiser(
                //    randomizer,
                //    false,
                //    new ElasticNoiser(randomizer, 1, 28, 28, true),
                //    new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, 784)),
                //    new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, 784)),
                //    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer, 784)),
                //    new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer, 784)),
                //    new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, 784))
                //    );

                
                var en = new ElasticNoiser(
                    randomizer, 
                    1, 
                    28, 
                    28, 
                    true);

                var spn = new SaltAndPepperNoiser(
                    randomizer,
                    .99f,
                    new RectangleRange(
                        randomizer,
                        28,
                        28,
                        new rint(randomizer, 6, 14),
                        new rint(randomizer, 6, 14),
                        false));

                var gn = new GaussNoiser(
                    0.2f,
                    false,
                    new RectangleRange(
                        randomizer,
                        28,
                        28,
                        new rint(randomizer, 6, 14),
                        new rint(randomizer, 6, 14),
                        false));

                var mn = new MultiplierNoiser(
                    randomizer,
                    1f,
                    new RectangleRange(
                        randomizer,
                        28,
                        28,
                        new rint(randomizer, 6, 14),
                        new rint(randomizer, 6, 14),
                        false));

                var dcn = new DistanceChangeNoiser(
                    randomizer, 
                    1f, 
                    3, 
                    new RectangleRange(
                        randomizer,
                        28,
                        28,
                        new rint(randomizer, 6, 14),
                        new rint(randomizer, 6, 14),
                        false));


                var zmn = new ZeroMaskingNoiser(
                    randomizer,
                    0.25f,
                    new RectangleRange(
                        randomizer,
                        28,
                        28,
                        new rint(randomizer, 6, 14),
                        new rint(randomizer, 6, 14),
                        false));


                var noiser = new SequenceNoiser(
                    randomizer,
                    false,
                    en,
                    gn,
                    mn,
                    dcn,
                    spn,
                    zmn
                    );


                var nv = new NoiserVisualizer(
                    randomizer,
                    noiser);

                var data = MNISTDataProvider.GetDataSet(
                    "_MNIST_DATABASE/mnist/trainingset/",
                    100
                    );
                data.Normalize();

                nv.Visuzalize(
                    data,
                    28,
                    28,
                    4
                    );

                return;
                //*/

                //TrainSDAE2D.DoTrain();
                //TrainAutoencoder2D.DoTrain();
                TrainMLPOnSDAE.DoTrain();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
