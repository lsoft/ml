using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
    public interface IDodfCalculator
    {
        float[] CalculateDodf(int a);
    }
}
