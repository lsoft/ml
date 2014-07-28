namespace MyNN.BeliefNetwork.ImageReconstructor
{
    public interface IDataArrayConverter
    {
        float[] Convert(float[] dataToConvert);
    }
}