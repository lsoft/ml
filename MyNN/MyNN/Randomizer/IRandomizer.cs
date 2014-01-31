namespace MyNN.Randomizer
{
    public interface IRandomizer
    {
        int Next(int maxValue);
        
        float Next();

        void NextBytes(byte[] buffer);
    }
}
