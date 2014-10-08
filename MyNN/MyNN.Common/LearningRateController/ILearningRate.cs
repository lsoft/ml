namespace MyNN.Common.LearningRateController
{
    public interface ILearningRate
    {
        float GetLearningRate(int epocheNumber);
    }
}
