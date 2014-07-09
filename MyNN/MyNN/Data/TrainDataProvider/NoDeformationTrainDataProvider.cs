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

        public bool IsClassificationAuencoderDataSet
        {
            get
            {
                return this._trainData.IsClassificationAuencoderDataSet;
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

        public DataSet GetDataSet(int epocheNumber)
        {
            return _trainData;
        }
    }
}
