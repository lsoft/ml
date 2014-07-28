using System;
using MyNN.BoltzmannMachines;

namespace MyNN.BeliefNetwork.ImageReconstructor
{
    public interface IStackedImageReconstructor : IImageReconstructor
    {
        void AddConverter(IDataArrayConverter converter);
    }
}