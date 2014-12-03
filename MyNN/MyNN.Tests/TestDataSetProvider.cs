using System;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSetProvider;

namespace MyNN.Tests
{
    internal class TestDataSetProvider : IDataSetProvider
    {
        private readonly IDataSet _dataSet;

        public int Count
        {
            get
            {
                return
                    _dataSet.Count;
            }
        }

        public TestDataSetProvider(
            IDataSet dataSet
            )
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }

            _dataSet = dataSet;
        }

        public IDataSet GetDataSet(int epochNumber)
        {
            return
                _dataSet;
        }
    }
}