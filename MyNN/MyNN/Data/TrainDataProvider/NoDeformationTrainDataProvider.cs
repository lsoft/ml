using System;

namespace MyNN.Data.TrainDataProvider
{
    public class NoDeformationTrainDataProvider : ITrainDataProvider
    {
        private readonly DataSet _trainData;

        public bool IsAuencoderDataSet
        {
            get
            {
                return this._trainData.IsAuencoderDataSet;
            }
        }

        public NoDeformationTrainDataProvider(DataSet trainData)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            _trainData = trainData;
        }

        public DataSet GetDeformationDataSet(int epocheNumber)
        {
            return _trainData;
        }
    }
}
