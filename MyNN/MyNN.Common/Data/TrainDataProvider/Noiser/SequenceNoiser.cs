using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser
{
    /// <summary>
    /// Применяет все нойзеры к каждому итему!
    /// Особенно имеет смысл, если у отдельных нойзеров рандомный IRange
    /// </summary>
    [Serializable]
    public class SequenceNoiser : INoiser
    {
        private readonly IRandomizer _randomizer;
        private readonly bool _isNeedToShuffle;
        private readonly INoiser[] _noiserList;

        public SequenceNoiser(
            IRandomizer randomizer,
            bool isNeedToShuffle,
            params INoiser[] noiserList)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (noiserList == null || noiserList.Length == 0 || noiserList.Any(j => j == null))
            {
                throw new ArgumentNullException("noiserList");
            }

            _randomizer = randomizer;
            _isNeedToShuffle = isNeedToShuffle;
            _noiserList = noiserList;
        }

        public float[] ApplyNoise(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            var length = data.Length;

            var result = new float[length];
            data.CopyTo(result, 0);

            var shuffledNoiserList = 
                _isNeedToShuffle 
                ? CreateShuffledNoiserList()
                : this._noiserList;

            foreach (var noiser in shuffledNoiserList)
            {
                result = noiser.ApplyNoise(result);
            }

            return
                result;
        }

        private INoiser[] CreateShuffledNoiserList()
        {
            var cloned = new List<INoiser>(this._noiserList);
            for (int i = 0; i < cloned.Count - 1; i++)
            {
                if (_randomizer.Next() > 0.5f)
                {
                    var newIndex = _randomizer.Next(cloned.Count);

                    var tmp = cloned[i];
                    cloned[i] = cloned[newIndex];
                    cloned[newIndex] = tmp;
                }
            }

            return
                cloned.ToArray();
        }

    }
}
