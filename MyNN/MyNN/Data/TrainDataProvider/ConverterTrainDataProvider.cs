using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data.DataSetConverter;

namespace MyNN.Data.TrainDataProvider
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
