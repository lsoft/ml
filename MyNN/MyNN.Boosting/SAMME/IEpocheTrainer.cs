using System.Collections.Generic;

namespace MyNN.Boosting.SAMME
{
    public interface IEpocheTrainer
    {
        IEpocheClassifier TrainEpocheClassifier(
            List<double[]> epocheInputs, 
            List<int> epocheLabels,
            int outputLength,
            int inputLength);
    }
}