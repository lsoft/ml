namespace MyNN.Boltzmann.BeliefNetwork.Accuracy
{
    public interface IAccuracyController
    {
        void AddError(
            int currentEpochNumber,
            float error
            );

        bool IsNeedToStop();

        bool IsLastEpochBetterThanPrevious();

        IAccuracyController Clone();
    }
}