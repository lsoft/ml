﻿namespace MyNN.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        IDataSet GetDataSet(int epocheNumber);
    }
}
