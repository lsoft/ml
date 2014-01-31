using System.Collections.Generic;
using MyNN.Data;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    /// <summary>
    /// Distances provider
    /// </summary>
    public interface IDistanceDictCalculator
    {
        /// <summary>
        /// Calculate distances
        /// </summary>
        /// <param name="fxwList">Input representation in floats</param>
        /// <returns>Distances</returns>
        DodfDistanceContainer CalculateDistances(List<DataItem> fxwList);
    }
}