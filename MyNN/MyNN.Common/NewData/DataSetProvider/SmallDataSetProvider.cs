using System;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Common.NewData.DataSetProvider
{
    [Serializable]
    public class SmallDataSetProvider : IDataSetProvider
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

        public SmallDataSetProvider(
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