﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Randomizer;

namespace MyNN.Data.TrainDataProvider.Noiser
{
    public interface INoiser
    {
        float[] ApplyNoise(float[] data);
    }

    //public class Shift2DNoiser : INoiser
    //{
    //    private readonly IRandomizer _randomizer;

    //    public Shift2DNoiser(
    //        IRandomizer randomizer,
    //        int imageWidth,
    //        int imageHeight,

    //        )
    //    {
    //        if (randomizer == null)
    //        {
    //            throw new ArgumentNullException("randomizer");
    //        }
    //        _randomizer = randomizer;
    //    }

    //    public float[] ApplyNoise(float[] data)
    //    {
    //        throw new NotImplementedException();
    //    }
    //}
}
