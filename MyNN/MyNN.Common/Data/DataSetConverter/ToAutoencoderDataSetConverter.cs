using System;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Dense;

namespace MyNN.Common.Data.DataSetConverter
{
    public class ToAutoencoderDataSetConverter : IDataSetConverter
    {
        private readonly IDataItemFactory _dataItemFactory;

        public ToAutoencoderDataSetConverter(
            IDataItemFactory dataItemFactory
            )
        {
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _dataItemFactory = dataItemFactory;
        }

        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var result =
                new DataSet(
                    beforeTransformation.Data.ConvertAll(j => _dataItemFactory.CreateDataItem(j.Input, j.Input))
                    );

            return result;
        }
    }
}