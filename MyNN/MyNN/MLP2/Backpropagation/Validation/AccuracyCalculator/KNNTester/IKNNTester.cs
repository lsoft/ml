using MyNN.MLP2.ForwardPropagation;

namespace MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator.KNNTester
{
    public interface IKNNTester
    {
        void Test(
            IForwardPropagation forwardPropagation,
            int takeIntoAccount,
            out int total,
            out int correct
            );
    }
}