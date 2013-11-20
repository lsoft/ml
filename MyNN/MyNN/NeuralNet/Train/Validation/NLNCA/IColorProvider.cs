using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace MyNN.NeuralNet.Train.Validation.NLNCA
{
    public interface IColorProvider
    {
        Color[] GetColors();
    }
}
