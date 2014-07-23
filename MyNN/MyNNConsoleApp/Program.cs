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
using MyNNConsoleApp.RefactoredForDI;

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

                TrainRBM.DoTrain();


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
