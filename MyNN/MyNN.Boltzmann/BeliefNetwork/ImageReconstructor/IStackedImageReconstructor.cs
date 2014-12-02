using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter;
using MyNN.Boltzmann.BoltzmannMachines;

namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor
{
    public interface IStackedImageReconstructor : IImageReconstructor
    {
        void AddConverter(IDataArrayConverter converter);
    }
}