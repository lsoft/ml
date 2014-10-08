using System;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4.Sorter
{
    public class SorterFactory<T> : ISorterFactory
        where T : ISorter
    {
        public ISorter CreateSorter(CLProvider clProvider)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            return
                (ISorter)Activator.CreateInstance(
                    typeof(T),
                    new object[]
                    {
                        clProvider
                    });
        }
    }
}