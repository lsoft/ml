using MyNN.MLP.ForwardPropagation;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.AccuracyCalculator.KNNTester
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