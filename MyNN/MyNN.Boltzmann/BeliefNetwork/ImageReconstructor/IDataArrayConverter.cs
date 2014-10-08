namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor
{
    public interface IDataArrayConverter
    {
        float[] Convert(float[] dataToConvert);
    }
}