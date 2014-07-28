using System;
using MyNN.BoltzmannMachines;

namespace MyNN.BeliefNetwork.ImageReconstructor
{
    public interface IDataArrayConverter
    {
        float[] Convert(float[] from);
    }

    public interface IStackedImageReconstructor : IImageReconstructor
    {
        void AddConverter(
            Func<float[], float[]> converter);
    }
}