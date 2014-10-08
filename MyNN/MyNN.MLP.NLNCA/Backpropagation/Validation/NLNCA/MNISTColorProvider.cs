using System.Drawing;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.NLNCA
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
