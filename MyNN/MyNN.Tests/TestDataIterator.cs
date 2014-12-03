using System;
using System.Collections;
using System.Collections.Generic;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet.Iterator;

namespace MyNN.Tests
{
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
}