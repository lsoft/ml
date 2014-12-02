using System;
using System.Collections.Generic;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;

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

            var list = new List<IDataItem>();

            foreach (var di in beforeTransformation.Data)
            {
                var newdi = _dataItemFactory.CreateDataItem(di.Input, di.Input);
                
                list.Add(newdi);
            }

            var result = new DataSet(list);

            //var result =
            //    new DataSet(
            //        beforeTransformation.Data.ConvertAll(j => _dataItemFactory.CreateDataItem(j.Input, j.Input))
            //        );

            return result;
        }
    }
}