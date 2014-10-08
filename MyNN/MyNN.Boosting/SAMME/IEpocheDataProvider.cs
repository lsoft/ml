using System.Collections.Generic;

namespace MyNN.Boosting.SAMME
{
    public interface IEpocheDataProvider
    {
        void GetEpocheDataSet(int epocheNumber, double[][] inputs, int[] labels, out List<double[]> epocheInputs, out List<int> epocheLabels, float[] w);
    }
}