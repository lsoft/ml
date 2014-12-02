namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter
{
    public interface IDataArrayConverter
    {
        float[] Convert(float[] dataToConvert);
    }
}