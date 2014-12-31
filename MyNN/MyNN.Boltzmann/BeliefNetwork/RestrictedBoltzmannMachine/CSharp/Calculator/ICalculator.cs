namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator
{
    public interface ICalculator
    {
        string VisibleFunctionName
        {
            get;
        }

        string HiddenFunctionName
        {
            get;
        }

        void SampleHidden(
            float[] weights,
            float[] hiddenBiases,
            float[] targetHidden,
            float[] fromVisible
            );

        void CalculateHidden(
            float[] weights,
            float[] hiddenBiases,
            float[] targetHidden,
            float[] fromVisible
            );

        void SampleVisible(
            float[] weights,
            float[] visibleBiases,
            float[] targetVisible,
            float[] fromHidden
            );

        void CalculateVisible(
            float[] weights,
            float[] visibleBiases,
            float[] targetVisible,
            float[] fromHidden
            );
    }
}