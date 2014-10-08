using System;
using System.Drawing;
using System.Linq;

namespace MyNN.Common.Data.Visualizer
{
    public class BitmapHelper
    {
        public static void CreateContrastEnhancedBitmapFrom(
            Bitmap bitmap,
            int left,
            int top,
            float[] layer)
        {
            var max = layer.Take(28 * 28).Max(val => val);// < 0 ? 0 : val);
            var min = layer.Take(28 * 28).Min(val => val);// < 0 ? 0 : val);

            if (Math.Abs(min - max) <= float.Epsilon)
            {
                min = 0;
                max = 1;
            }

            for (int x = 0; x < 28; x++)
            {
                for (int y = 0; y < 28; y++)
                {
                    var value = layer[PointToIndex(x, y, 28)];
                    value = (value - min) / (max - min);
                    var b = (byte)Math.Max(0, Math.Min(255, value * 255.0));

                    bitmap.SetPixel(left + x, top + y, Color.FromArgb(b, b, b));
                }
            }
        }

        private static int PointToIndex(int x, int y, int width)
        {
            return y * width + x;
        }

    }
}
