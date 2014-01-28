using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaForwardOptimizer
    {
        public static void Optimize()
        {

            var rndSeed = 1;
            var randomizer = new DefaultRandomizer(ref rndSeed);
            
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                100
                );
            trainData.Data.ForEach(item => item.Input.Transform((value) => value + 0.1f)); //чтобы не было нулей в датасете, а то вдруг алгоритм "забывает" например учесть последний флоат в датаитеме...
            //trainData.Normalize();
            trainData = trainData.ConvertToAutoencoder();

            var mlp = new MLP(
                randomizer,
                null,
                null,
                new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f),
                        new SigmoidFunction(1f),
                    },
                new[]
                    {
                        784,
                        784 * 10,
                        784
                    });

           var nvidiaResults = ProfileNvidiaGPU(
                randomizer,
                trainData,
                mlp);

           var intelResults = ProfileIntelCPU(
               randomizer,
               trainData,
               mlp);

           var nvidiaSum = nvidiaResults.Sum(j => j.State.Sum());
           var intelSum = intelResults.Sum(j => j.State.Sum());

           Console.WriteLine(
               "DIFF = {0}",
               (nvidiaSum - intelSum));
        }

        private static List<ILayerState> ProfileNvidiaGPU(
            DefaultRandomizer randomizer,
            DataSet trainData,
            MLP mlp)
        {
            using (var clProvider = new CLProvider(
                new NvidiaOrAmdGPUDeviceChooser(),
                true))
            {

                var forward = new GPUForwardPropagation(
                    mlp,
                    clProvider);

                TimeSpan propagationTime;
                var results = forward.ComputeOutput(trainData, out propagationTime);

                ConsoleAmbientContext.Console.WriteLine(
                    "NVIDIA GPU takes {0}",
                    propagationTime);

                return results;
            }
        }
        
        private static List<ILayerState> ProfileIntelCPU(
            DefaultRandomizer randomizer,
            DataSet trainData,
            MLP mlp)
        {
            using (var clProvider = new CLProvider(
                new IntelCPUDeviceChooser(),
                true))
            {
                var forward = new CPUForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    mlp,
                    clProvider);

                TimeSpan propagationTime;
                var results = forward.ComputeOutput(trainData, out propagationTime);

                ConsoleAmbientContext.Console.WriteLine(
                    "INTEL CPU takes {0}",
                    propagationTime);

                return results;
            }
        }
    }
}
