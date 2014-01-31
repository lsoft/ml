using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.MLP2.Transposer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNNConsoleApp.TransposeExperiments
{

    public class Transpose0
    {
        public static void Execute()
        {
            for(var width = 16; width < 17; width++)
            {
                for (var height = 784; height < 785; height++)
                {
                    Console.WriteLine("w = {0},  h = {1}", width, height);

                    using (var clProvider = new CLProvider())
                    {
                        var source = clProvider.CreateFloatMem(
                            width*height,
                            MemFlags.CopyHostPtr | MemFlags.ReadOnly);

                        for (var cc = 0; cc < source.Array.Length; cc++)
                        {
                            source.Array[cc] = cc;
                        }

                        source.Write(BlockModeEnum.Blocking);

                        var t = new TransposerNvidia(
                            clProvider,
                            source,
                            width,
                            height);

                        t.Transpose();

                        clProvider.QueueFinish();

                        t.Destination.Read(BlockModeEnum.Blocking);

                        ConsoleDump("SOURCE:", source.Array, width, height);
                        ConsoleDump("TRANSP:", t.Destination.Array, height, width);

                        CheckEquals(source.Array, t.Destination.Array, width, height);
                    }
                }
            }


            Console.WriteLine("Transpose 0 finished!");
            Console.ReadLine();
        }

        private static void CheckEquals(
            float[] source,
            float[] transposed,
            int width,
            int height)
        {
            if (source == null)
            {
                throw new ArgumentNullException("source");
            }
            if (transposed == null)
            {
                throw new ArgumentNullException("transposed");
            }

            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    var s = source[h*width + w];
                    var d = transposed[w * height + h];

                    var diff = s - d;

                    if (Math.Abs(diff) >= float.Epsilon)
                    {
                        throw new Exception(diff.ToString());
                    }
                }
            }
        }


        public static void ConsoleDump(string name, float[] body, int width, int height)
        {
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (body == null)
            {
                throw new ArgumentNullException("body");
            }

            Console.WriteLine(name);

            for (var h = 0; h < height; h++)
            {
                var listw = new List<float>();
                for (var w = 0; w < width; w++)
                {
                    listw.Add(body[h * width + w]);
                }

                var s = string.Join(
                    " ",
                    listw.ConvertAll(j => j.ToString("0000")).ToArray());

                Console.WriteLine(s);
            }
        }
    }
}
