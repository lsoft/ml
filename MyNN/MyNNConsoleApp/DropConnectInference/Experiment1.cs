using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Text;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment1
    {
        public static void Execute()
        {
            var rndSeed = 234;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var wv =
                new[]
                    {
                        0.2f,
                        0.3f,
                        0.4f,
                        0.5f,
                        0.6f,
                        0.2f,
                    };

            const float p = 0.5f;

            const int sampleCount = 100000;

            var prevLayer = new MLPLayer(
                wv.Length - 1,
                true);

            var currentLayer = new MLPLayer(
                new LinearFunction(1f),
                1,
                prevLayer.NonBiasNeuronCount,
                false,
                true,
                randomizer);


            {
                var wvbino = new WVBino(p, wv.Length, wv);

                var before_cs = DateTime.Now;

                var u_bino_sum = 0.0;
                for (var sumindex = 0; sumindex < sampleCount; sumindex++)
                {
                    u_bino_sum += wvbino.GetU();
                }

                var after_cs = DateTime.Now;

                Console.WriteLine("C# Binomial Inferece = {0}", u_bino_sum / (double)sampleCount);
                Console.WriteLine("It takes {0}", (after_cs - before_cs));
            }


            using (var clProvider = new CLProvider())
            {
                var prevLayerStateMem = clProvider.CreateFloatMem(
                    wv.Length,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                for (var cc = 0; cc < prevLayerStateMem.Array.Length; cc++)
                {
                    prevLayerStateMem.Array[cc] = 1f;
                }
                prevLayerStateMem.Write(BlockModeEnum.Blocking);

                var weightMem = clProvider.CreateFloatMem(
                    wv.Length,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                wv.CopyTo(weightMem.Array, 0);
                weightMem.Write(BlockModeEnum.Blocking);

                var currentLayerStateMem = clProvider.CreateFloatMem(
                    1,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                var inf1 = new OpenCLLayerInference(
                    randomizer,
                    clProvider,
                    sampleCount,
                    prevLayer,
                    currentLayer,
                    weightMem,
                    prevLayerStateMem,
                    currentLayerStateMem,
                    p);

                var before_cl = DateTime.Now;
                
                inf1.InferenceLayer();

                var after_cl = DateTime.Now;

                currentLayerStateMem.Read(BlockModeEnum.Blocking);

                Console.WriteLine("OpenCL Guassian inference: {0}", currentLayerStateMem.Array[0]);
                Console.WriteLine("It takes {0}", (after_cl - before_cl));

            }

                Console.WriteLine("Experiment #1 finished");
                Console.ReadLine();
        }
    }
}
