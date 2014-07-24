namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container
{
    public interface IContainer
    {
        int VisibleNeuronCount
        {
            get;
        }

        int HiddenNeuronCount
        {
            get;
        }

        void SetTrainItem(float[] input);

        void ClearNabla();

        void CalculateNabla();

        void UpdateWeights(float learningRate);

        float GetError();
    }
}