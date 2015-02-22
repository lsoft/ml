using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNNConsoleApp.Conv
{
    public class LayerVisualizer
    {
        public static void Show(
            string header,
            float[] data,
            int width,
            int height
            )
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }
            if (data.Length != width * height)
            {
                throw new ArgumentException("data.Length != width*height");
            }

            var colors = new ConsoleColor[]
            {
                ConsoleColor.DarkBlue,
                ConsoleColor.Blue,
                ConsoleColor.Green,
                ConsoleColor.Yellow,
                ConsoleColor.White
            };

            Console.ResetColor();
            Console.WriteLine(header + ":");

            var step = 1.01f / colors.Length;

            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    var v = data[h * width + w];

                    var vs = v.ToString("N2");
                    var toout = vs.PadLeft(6);

                    if (v < 0)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkBlue;
                    }
                    else
                    {
                        var colorIndex = (int)(v / step);

                        if (colorIndex >= colors.Length)
                        {
                            colorIndex = colors.Length - 1;
                        }

                        Console.ForegroundColor = colors[colorIndex];
                    }

                    Console.Write(toout);
                    Console.Write("  ");
                }

                Console.WriteLine(string.Empty);
            }

            Console.WriteLine(string.Empty);
            Console.ResetColor();


        }
    }
}
