using System.Collections.Generic;
using MyNN.Data;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public interface IDistanceDictFactory
    {
        Dictionary<int, float[]> CreateDistanceDict(List<DataItem> fxwList);
    }
}