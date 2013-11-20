using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace MyNN.NeuralNet.Train.Validation.NLNCA
{
    public class MNISTColorProvider : IColorProvider
    {
        public Color[] GetColors()
        {
            return new Color[]
                             {
                                 Color.Red,
                                 Color.Green,
                                 Color.Blue,
                                 Color.YellowGreen,
                                 Color.Black,

                                 Color.Orange,
                                 Color.MediumSlateBlue,
                                 Color.Turquoise,
                                 Color.OliveDrab,
                                 Color.Gray
                             };
        }
    }
}
