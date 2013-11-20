using System;
using System.Collections.Generic;

namespace MyNN.Boosting.SAMMEBoosting.DataProviders
{
    public class RandomSeventyFivePercentDataProvider : IEpocheDataProvider
    {
        public RandomSeventyFivePercentDataProvider()
        {
        }

        public void GetEpocheDataSet(int epocheNumber, double[][] inputs, int[] labels, out List<double[]> epocheInputs, out List<int> epocheLabels, float[] w)
        {
            epocheInputs = new List<double[]>();
            epocheLabels = new List<int>();

            //рандомно формируем поднабор
            var rnd = new Random(epocheNumber);
            for (var cc = 0; cc < inputs.Length; cc++)
            {
                if (rnd.NextDouble() < 0.75)
                {
                    epocheInputs.Add(inputs[cc]);
                    epocheLabels.Add(labels[cc]);
                }
            }

            //рандомизируем выборку
            this.Shuffle(epocheInputs, epocheLabels, rnd);
        }

        private void Shuffle(
            List<double[]> epocheInputs,
            List<int> epocheLabels,
            Random rnd)
        {
            for (int i = 0; i < epocheInputs.Count - 1; i++)
            {
                if (rnd.NextDouble() >= 0.5d)
                {
                    var newIndex = rnd.Next(epocheInputs.Count);

                    var tmp = epocheInputs[i];
                    epocheInputs[i] = epocheInputs[newIndex];
                    epocheInputs[newIndex] = tmp;

                    var tmp2 = epocheLabels[i];
                    epocheLabels[i] = epocheLabels[newIndex];
                    epocheLabels[newIndex] = tmp2;
                }
            }
        }
    }
}