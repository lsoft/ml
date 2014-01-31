using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment3
    {
        public static void Execute()
        {
            var rndSeed = 234;
            var randomizer = new DefaultRandomizer(ref rndSeed);

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

            const float p = 0.8f;

            const int sampleCount = 100000;

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
                var wvnormal_mean = 0.0;
                for (var cc = 0; cc < wv.Length; cc++)
                {
                    wvnormal_mean += wv[cc];
                }
                wvnormal_mean *= p;

                var wvnormal_var = 0.0;
                for (var cc = 0; cc < wv.Length; cc++)
                {
                    wvnormal_var += wv[cc] * wv[cc];
                }
                wvnormal_var *= p * (1.0f - p);

                var wvnorma = new MyNormal(wvnormal_mean, Math.Sqrt(wvnormal_var));

                var before_cs = DateTime.Now;

                var u_norma_sum = 0.0;
                for (var sumindex = 0; sumindex < sampleCount; sumindex++)
                {
                    u_norma_sum += wvnorma.Sample();
                }

                var after_cs = DateTime.Now;

                Console.WriteLine("--- normal median = {0} ----", wvnormal_mean);
                Console.WriteLine("C# Normal Inference = {0}", u_norma_sum / (double)sampleCount);
                Console.WriteLine("It takes {0}", (after_cs - before_cs));
            }

            {
                var folderName = "_DCMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var mlp = new MLP(
                    randomizer,
                    null,
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
                    var forward = new InferenceOpenCLForwardPropagation<OpenCLLayerInference>(
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

            {
                var folderName = "_DCMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var mlp = new MLP(
                    randomizer,
                    null,
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
                    var forward = new InferenceOpenCLForwardPropagation<OpenCLLayerInferenceNew>(
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

                    Console.WriteLine("Drop connect forward inference NEW: {0}", result[0].State[0]);
                    Console.WriteLine("It takes {0}", (after_cl - before_cl));

                }
            }

            {
                var folderName = "_DCMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var mlp = new MLP(
                    randomizer,
                    null,
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
                    var forward = new InferenceOpenCLForwardPropagation<OpenCLLayerInferenceNew16>(
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

                    Console.WriteLine("Drop connect forward inference NEW16: {0}", result[0].State[0]);
                    Console.WriteLine("It takes {0}", (after_cl - before_cl));

                }
            }

            //{
            //    var folderName = "_DCMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

            //    var mlp = new MLP(
            //        randomizer,
            //        null,
            //        folderName,
            //        new IFunction[]
            //        {
            //            null,
            //            new LinearFunction(1f)
            //        },
            //        new int[]
            //        {
            //            wv.Length - 1,
            //            1
            //        });

            //    wv.CopyTo(mlp.Layers[1].Neurons[0].Weights, 0);

            //    using (var clProvider = new CLProvider())
            //    {
            //        var forward = new InferenceOpenCLForwardPropagation<OpenCLLayerInferenceMedian>(
            //            VectorizationSizeEnum.VectorizationMode16,
            //            mlp,
            //            clProvider,
            //            randomizer,
            //            sampleCount,
            //            p
            //            );

            //        var dataset = new DataSet(
            //            new List<DataItem>()
            //            {
            //                new DataItem(
            //                    new float[5]
            //                    {
            //                        1f, 1f, 1f, 1f, 1f
            //                    }, 
            //                    new[] {0f})
            //            });

            //        var before_cl = DateTime.Now;

            //        var result = forward.ComputeOutput(dataset);

            //        var after_cl = DateTime.Now;

            //        Console.WriteLine("Drop connect forward inference MEDIAN: {0}", result[0].State[0]);
            //        Console.WriteLine("It takes {0}", (after_cl - before_cl));

            //    }
            //}


            
            Console.WriteLine("Experiment #3 finished");
            Console.ReadLine();
        }
    }
}
