using System;
using System.Collections.Generic;
using System.Linq;

namespace MyNN.Boosting.SAMMEBoosting
{
    public class SAMMEClassifierSet
    {
        private readonly List<IEpocheClassifier> _classifiers;
        private readonly List<float> _alphas;

        public SAMMEClassifierSet()
        {
            _classifiers = new List<IEpocheClassifier>();
            _alphas = new List<float>();
        }

        public SAMMEClassifierSet(SAMMEClassifierSet o)
        {
            _classifiers = new List<IEpocheClassifier>(o._classifiers);
            _alphas = new List<float>(o._alphas);
        }

        public int Classify(float[] data, int outputLength)
        {
            var doubleData = data.ToList().ConvertAll(j => (double)j).ToArray();

            var votes = new float[outputLength];
            for (var cc = 0; cc < _classifiers.Count; cc++)
            {
                var k = _classifiers[cc].Compute(doubleData);
                votes[k] += _alphas[cc];
            }

            var maxV = votes.Max();
            var countMaxV = votes.Count(j => Math.Abs(j - maxV) < float.Epsilon);

            var result = -1;

            if (countMaxV == 1)
            {
                result = votes.ToList().IndexOf(maxV);
            }

            return result;
        }

        public void Add(IEpocheClassifier classifier, float alpha)
        {
            _classifiers.Add(classifier);
            _alphas.Add(alpha);
        }
    }
}