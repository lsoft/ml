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
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2;

using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
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

                //TrainRBM.DoTrainBB();
                //TrainRBM.DoTrainLNRELU();

                //TrainDBN.DoTrainBB();
                TrainDBN.DoTrainLNRELU();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
