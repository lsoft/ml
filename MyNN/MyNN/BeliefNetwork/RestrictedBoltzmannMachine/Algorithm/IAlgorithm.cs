using System.Collections.Generic;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm
{
    public interface IAlgorithm
    {
        string Name
        {
            get;
        }

        void PrepareTrain(
            int batchSize);

        void PrepareBatch();

        void CalculateSamples(
            int indexIntoBatch,
            int maxGibbsChainLength
            );

        void BatchFinished();

        float[] CalculateReconstructed();
        
        ICollection<float[]> GetFeatures();
    }
}
