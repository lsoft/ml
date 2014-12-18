using System.Collections.Generic;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
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
        DodfDistanceContainer CalculateDistances(List<IDataItem> fxwList);
    }
}