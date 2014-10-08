using System.Collections.Generic;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm
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

        void ExecuteGibbsSampling(
            int indexIntoBatch,
            int maxGibbsChainLength
            );

        void BatchFinished();

        //result should not contains bias value!
        float[] CalculateVisible();

        //result should not contains bias value!
        float[] CalculateHidden();

        //result should not contains bias value!
        float[] CalculateReconstructed();
        
        ICollection<float[]> GetFeatures();
    }
}
