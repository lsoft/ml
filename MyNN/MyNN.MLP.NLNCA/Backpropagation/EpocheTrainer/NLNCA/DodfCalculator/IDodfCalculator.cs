namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator
{
    /// <summary>
    /// dOdF calculator.
    /// For details about dOdF please refer https://www.cs.toronto.edu/~hinton/absps/nonlinnca.pdf
    /// </summary>
    public interface IDodfCalculator
    {
        /// <summary>
        /// Calculate dOdF for train data item
        /// </summary>
        /// <param name="a">Train data item index</param>
        /// <returns>dOdF coefs</returns>
        float[] CalculateDodf(int a);
    }
}
