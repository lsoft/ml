using System;
using System.Linq;
using MyNN.Common.Other;

namespace MyNN.Common.Data.TrainDataProvider.Noiser
{
    /// <summary>
    /// Применяет все нойзеры к каждому итему методом параллельного вызова всех
    /// дочерних нойзеров и усреднения!
    /// Особенно имеет смысл, если у отдельных нойзеров рандомный IRange
    /// </summary>
    [Serializable]
    public class AverageNoiser : INoiser
    {
        private readonly INoiser[] _noiserList;

        public AverageNoiser(
            params INoiser[] noiserList)
        {
            if (noiserList == null || noiserList.Length == 0 || noiserList.Any(j => j == null))
            {
                throw new ArgumentNullException("noiserList");
            }

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

            var beforenoiser = new float[length];

            foreach (var noiser in this._noiserList)
            {
                data.CopyTo(beforenoiser, 0);

                var afternoiser = noiser.ApplyNoise(beforenoiser);

                result.Transform((index, orig) => orig + afternoiser[index]);
            }

            result.Transform(a => a / this._noiserList.Length);

            return
                result;
        }

    }
}
