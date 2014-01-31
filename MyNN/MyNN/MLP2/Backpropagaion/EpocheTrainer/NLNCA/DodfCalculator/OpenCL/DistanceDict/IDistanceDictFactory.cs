using System.Collections.Generic;
using MyNN.Data;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public interface IDistanceDictFactory
    {
        DodfDictionary CreateDistanceDict(List<DataItem> fxwList);
    }
}