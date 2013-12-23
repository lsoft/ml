using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public interface INoiser
    {
        float[] ApplyNoise(float[] data);
    }
}
