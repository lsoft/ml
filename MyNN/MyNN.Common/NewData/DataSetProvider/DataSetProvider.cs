using System;
using System.Runtime.InteropServices;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;

namespace MyNN.Common.NewData.DataSetProvider
{
    [Serializable]
    public class DataSetProvider : IDataSetProvider
    {
        private readonly IDataSetFactory _dataSetFactory;
        private readonly IDataItemLoader _itemLoader;

        public int Count
        {
            get
            {
                return
                    this._itemLoader.Count;
            }
        }

        public DataSetProvider(
            IDataSetFactory dataSetFactory,
            IDataItemLoader itemLoader
            )
        {
            if (dataSetFactory == null)
            {
                throw new ArgumentNullException("dataSetFactory");
            }
            if (itemLoader == null)
            {
                throw new ArgumentNullException("itemLoader");
            }

            _dataSetFactory = dataSetFactory;
            _itemLoader = itemLoader;
        }


        public IDataSet GetDataSet(int epochNumber)
        {
            return 
                _dataSetFactory.CreateDataSet(
                    _itemLoader,
                    epochNumber
                    );
        }
    }
}
