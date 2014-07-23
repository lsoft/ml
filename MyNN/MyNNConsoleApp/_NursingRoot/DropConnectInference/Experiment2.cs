﻿using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment2
    {
        public static void Execute()
        {
            var rndSeed = 234;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var wv =
                new[]
                    {
                        0.9f,
                        0.3f,
                        0.1f,
                        0.5f,
                        0.13f,
                        0.25f,
                    };

            const float p = 0.2f;

            const int sampleCount = 100000;

            //var prevLayer = new Layer(
            //    wv.Length - 1,
            //    true);

            //var currentLayer = new Layer(
            //    new LinearFunction(1f),
            //    1,
            //    prevLayer.NonBiasNeuronCount,
            //    false,
            //    true,
            //    randomizer);


            {
                var wvbino = new WVBino(p, wv.Length, wv);

                var before_cs = DateTime.Now;

                var u_bino_sum = 0.0;
                for (var sumindex = 0; sumindex < sampleCount; sumindex++)
                {
                    u_bino_sum += wvbino.GetU();
                }

                var after_cs = DateTime.Now;

                Console.WriteLine("C# Binomial Inference = {0}", u_bino_sum / (double)sampleCount);
                Console.WriteLine("It takes {0}", (after_cs - before_cs));
            }

            {
                var folderName = "_DCMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

                var mlpf = new MLPFactory(
                    layerFactory);

                var mlp = mlpf.CreateMLP(
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new LinearFunction(1f)
                    },
                    new int[]
                    {
                        wv.Length - 1,
                        1
                    });

                wv.CopyTo(mlp.Layers[1].Neurons[0].Weights, 0);

                using (var clProvider = new CLProvider())
                {
                    var forward = new InferenceOpenCLForwardPropagation<NaiveLayerInference>(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        clProvider,
                        randomizer,
                        sampleCount,
                        p
                        );

                    var dataset = new DataSet(
                        new List<DataItem>()
                        {
                            new DataItem(
                                new float[5]
                                {
                                    1f, 1f, 1f, 1f, 1f
                                }, 
                                new[] {0f})
                        });

                    var before_cl = DateTime.Now;

                    var result = forward.ComputeOutput(dataset);

                    var after_cl = DateTime.Now;

                    Console.WriteLine("Drop connect forward inference: {0}", result[0].State[0]);
                    Console.WriteLine("It takes {0}", (after_cl - before_cl));

                }
            }

            Console.WriteLine("Experiment #2 finished");
            Console.ReadLine();
        }
    }
}