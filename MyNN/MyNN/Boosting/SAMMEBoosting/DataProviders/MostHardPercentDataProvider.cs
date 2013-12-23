using System;
using System.Collections.Generic;
using System.IO;

namespace MyNN.Boosting.SAMMEBoosting.DataProviders
{
    public class MostHardPercentDataProvider : IEpocheDataProvider
    {
        private readonly int _percentCount;

        public MostHardPercentDataProvider(int percentCount)
        {
            if (percentCount <= 0 || percentCount >= 100)
            {
                throw new ArgumentException("percentCount");
            }

            _percentCount = percentCount;
        }

        public void GetEpocheDataSet(int epocheNumber, double[][] inputs, int[] labels, out List<double[]> epocheInputs, out List<int> epocheLabels, float[] w)
        {
            epocheInputs = new List<double[]>();
            epocheLabels = new List<int>();

            //сортируем выборку по весу
            var indexes = new int[inputs.Length];
            for (var ii = 0; ii < indexes.Length; ii++)
            {
                indexes[ii] = ii;
            }

            var keys0 = (float[]) w.Clone();
            Array.Sort(keys0, indexes);

            //берем 75% весов (последних, с бОльшим весом)
            for (var cc = indexes.Length - 1; cc >= (indexes.Length * (100 - _percentCount) / 100); cc--)
            {
                epocheInputs.Add(inputs[indexes[cc]]);
                epocheLabels.Add(labels[indexes[cc]]);
            }
            //рандомизируем выборку
            this.Shuffle(epocheInputs, epocheLabels, new Random(epocheNumber));
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