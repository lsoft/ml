using System;

namespace MyNN.Data.TrainDataProvider
{
    public class NoDeformationTrainDataProvider : ITrainDataProvider
    {
        private readonly IDataSet _trainData;

        public bool IsAuencoderDataSet
        {
            get
            {
                return this._trainData.IsAuencoderDataSet;
            }
        }

        public NoDeformationTrainDataProvider(IDataSet trainData)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }

            _trainData = trainData;
        }

        public IDataSet GetDataSet(int epocheNumber)
        {
            return _trainData;
        }
    }
}
