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

    internal class TestDataIterator : IDataIterator
    {
        private readonly IList<IDataItem> _dataList;
        private int _currentIndex;

        public IDataItem Current
        {
            get
            {
                if (_currentIndex < 0)
                {
                    return
                        null;
                }

                return
                    _dataList[_currentIndex];
            }
        }

        object IEnumerator.Current
        {
            get
            {
                return Current;
            }
        }

        public int Count
        {
            get
            {
                return
                    _dataList.Count;
            }
        }

        public TestDataIterator(
            IList<IDataItem> dataList
            )
        {
            if (dataList == null)
            {
                throw new ArgumentNullException("dataList");
            }
            _dataList = dataList;
        }


        public bool MoveNext()
        {
            if (_currentIndex < _dataList.Count - 1)
            {
                _currentIndex++;

                return true;
            }

            return
                false;
        }

        public void Reset()
        {
            _currentIndex = 0;
        }

        public void Dispose()
        {
            //nothing to do
        }

    }

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
