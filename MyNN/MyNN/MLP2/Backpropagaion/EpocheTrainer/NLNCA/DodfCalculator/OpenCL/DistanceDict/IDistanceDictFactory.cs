using System.Collections.Generic;
using MyNN.Data;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public interface IDistanceDictFactory
    {
        Dictionary<int, float[]> CreateDistanceDict(List<DataItem> fxwList);
    }
}