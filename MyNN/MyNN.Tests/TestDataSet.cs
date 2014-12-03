using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;

namespace MyNN.Tests
{
    internal class TestDataSet : IDataSet
    {
        private readonly IList<IDataItem> _dataList;

        public int Count
        {
            get
            {
                return
                    _dataList.Count;
            }
        }

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    _dataList[0].InputLength == _dataList[0].OutputLength;
            }
        }

        public int InputLength
        {
            get
            {
                return
                    _dataList[0].InputLength;
            }
        }

        public int OutputLength
        {
            get
            {
                return
                    _dataList[0].OutputLength;
            }
        }

        public TestDataSet(
            IList<IDataItem> dataList
            )
        {
            if (dataList == null)
            {
                throw new ArgumentNullException("dataList");
            }
            _dataList = dataList;
        }

        public IDataIterator StartIterate()
        {
            return 
                new TestDataIterator(_dataList);
        }

        public IEnumerator<IDataItem> GetEnumerator()
        {
            foreach (var i in _dataList)
            {
                yield return i;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

    }
}
