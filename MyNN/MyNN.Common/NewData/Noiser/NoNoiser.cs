using System;

namespace MyNN.Common.NewData.Noiser
{
    [Serializable]
    public class NoNoiser : INoiser
    {
        public float[] ApplyNoise(float[] data)
        {
            var newData = new float[data.Length];
            data.CopyTo(newData, 0);

            return newData;
        }
    }
}