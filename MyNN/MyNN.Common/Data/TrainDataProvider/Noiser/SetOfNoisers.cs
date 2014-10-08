using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser
{
    /// <summary>
    /// Применяет случайный нойзер к каждому итему
    /// </summary>
    public class SetOfNoisers : INoiser
    {
        private readonly IRandomizer _randomizer;

        private class ProbabilityNoiserContainer
        {
            public float From
            {
                get;
                private set;
            }

            public float To
            {
                get;
                private set;
            }

            public float Distance
            {
                get;
                private set;
            }

            public INoiser Noiser
            {
                get;
                private set;
            }

            public ProbabilityNoiserContainer(float @from, float to, float distance, INoiser noiser)
            {
                if (noiser == null)
                {
                    throw new ArgumentNullException("noiser");
                }

                From = @from;
                To = to;
                Distance = distance;
                Noiser = noiser;
            }

            public bool IsHist(float randomValue)
            {
                return
                    this.From <= randomValue && randomValue < this.To;
            }
        }

        private readonly List<ProbabilityNoiserContainer> _probabilities;

        public SetOfNoisers(
            IRandomizer randomizer,
            params Pair<float, INoiser>[] noiserList)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (noiserList == null || noiserList.Length == 0 || noiserList.Any(j => j == null))
            {
                throw new ArgumentNullException("noiserList");
            }
            if (Math.Abs(noiserList.Sum(k => k.First) - 1.0f) > float.Epsilon)
            {
                throw new ArgumentException("Sum of probabilities must be equal 1");
            }

            _randomizer = randomizer;

            _probabilities = new List<ProbabilityNoiserContainer>();
            var @from = 0f;
            foreach (var n in noiserList)
            {
                _probabilities.Add(
                    new ProbabilityNoiserContainer(
                        @from,
                        @from + n.First,
                        n.First,
                        n.Second));

                @from += n.First;
            }
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var rndValue = _randomizer.Next();

            var n = this._probabilities.FirstOrDefault(j => j.IsHist(rndValue));

            if (n == null)
            {
                n = this._probabilities[0];
            }

            return
                n.Noiser.ApplyNoise(data);
        }
    }
}
