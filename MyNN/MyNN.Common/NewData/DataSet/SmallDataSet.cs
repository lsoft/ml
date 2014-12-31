using System;
using System.Collections;
using System.Collections.Generic;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet
{
    [Serializable]
    public class SmallDataSet : IDataSet
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

        public SmallDataSet(
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
                new SmallDataIterator(_dataList);
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


        internal class SmallDataIterator : IDataIterator
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

            public SmallDataIterator(
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
}