using System;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;

namespace MyNN.Common.Data.TrainDataProvider
{
    public class ConverterTrainDataProvider : ITrainDataProvider
    {
        private readonly IDataSetConverter _dataSetConverter;
        private readonly ITrainDataProvider _trainDataProvider;

        public ConverterTrainDataProvider(
            IDataSetConverter dataSetConverter,
            ITrainDataProvider trainDataProvider
            )
        {
            if (dataSetConverter == null)
            {
                throw new ArgumentNullException("dataSetConverter");
            }
            if (trainDataProvider == null)
            {
                throw new ArgumentNullException("trainDataProvider");
            }

            _dataSetConverter = dataSetConverter;
            _trainDataProvider = trainDataProvider;
        }

        public IDataSet GetDataSet(int epocheNumber)
        {
            var ds = _trainDataProvider.GetDataSet(epocheNumber);
            var sh = _dataSetConverter.Convert(ds);

            return sh;
        }
    }
}
