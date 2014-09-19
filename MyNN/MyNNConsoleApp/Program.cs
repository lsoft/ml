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
using MyNN.MLP2.ForwardPropagation.Classic;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
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

                //TrainSDAE2D.DoTrain();
                //TrainAutoencoder2D.DoTrain();
                //TrainMLPOnSDAE.DoTrain();

                using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
                {
                    var randomizer = new DefaultRandomizer(1098);

                    var dil = new List<DataItem>();
                    for (var cc = 0; cc < 100; cc++)
                    {
                        var iff = new float[5000];
                        iff.Fill(randomizer.Next());

                        var off = new float[] {0f};

                        var di = new DataItem(
                            iff,
                            off);

                        dil.Add(di);
                    }

                    var dataset = new DataSet(dil);

                    var mlpf = new MLPFactory(
                        new LayerFactory(
                            new NeuronFactory(
                                randomizer)));

                    var mlp = mlpf.CreateMLP(
                        DateTime.Now.ToString("yyyyMMddHHmmss"),
                        new IFunction[]
                        {
                            null,
                            new LinearFunction(1f),
                            new LinearFunction(1f),
                            new LinearFunction(1f),
                            new LinearFunction(1f),
                            new LinearFunction(1f),
                        },
                                new int[]
                        {
                            5000,
                            5000,
                            5000,
                            5000,
                            5000,
                            5000
                        });

                    var pcc = new MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU.PropagatorComponentConstructor(
                        clProvider
                        );

                    ILayerContainer[] containers;
                    ILayerPropagator[] propagators;
                    pcc.CreateComponents(
                        mlp,
                        out containers,
                        out propagators);

                    var forward = 
                        new ForwardPropagation2(
                            containers,
                            propagators,
                            mlp
                            );

                    var before = DateTime.Now;

                    var output = forward.ComputeOutput(dataset);

                    var after = DateTime.Now;

                    Console.WriteLine((after - before));

                    /*
                    var img0 = clProvider.CreateImg(
                        7,
                        7,
                        MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                    img0.Array.Fill((int i) => (float)i);

                    img0.Write(BlockModeEnum.Blocking);

                    var mem = clProvider.CreateFloatMem(
                        1,
                        MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                    mem.Write(BlockModeEnum.Blocking);

                    var img1 = clProvider.CreateImg(
                        1,
                        1,
                        MemFlags.CopyHostPtr | MemFlags.ReadWrite);


                    var k = clProvider.CreateKernel(@"
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void KernelImg(
    read_only image2d_t img0,
    __global float* mem,
    write_only image2d_t img1
    )
{
    int2 coord = (int2)(1, 1);
    float4 pixel = read_imagef(img0, sampler, coord);
    
    float newvalue = pixel.s0 + 5;

    mem[0] = newvalue;
    write_imagef(img1, (int2)(0,0), (float4)(newvalue, newvalue, newvalue, newvalue));
}
", "KernelImg");

                    k
                        .SetKernelArgImg(0, img0)
                        .SetKernelArgMem(1, mem)
                        .SetKernelArgImg(2, img1)
                        .EnqueueNDRangeKernel(
                            new int[]
                            {
                                1
                            });

                    clProvider.QueueFinish();

                    mem.Read(BlockModeEnum.Blocking);
                    img1.Read(BlockModeEnum.Blocking);
                    //*/

                    Console.ReadLine();
                }

                return;

                TestBackProp.DoTest();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
