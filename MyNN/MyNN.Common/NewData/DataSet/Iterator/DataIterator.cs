using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public class DataIterator : IDataIterator
    {
        private readonly IDataItemLoader _itemLoader;
        private readonly Func<IDataItemTransformation> _transformationFactory;

        private IDataItemTransformation _dataItemTransformation;
        private int _currentIndex;

        public IDataItem Current
        {
            get;
            private set;
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
                    _itemLoader.Count;
            }
        }

        public int InputLength
        {
            get;
            private set;
        }

        public DataIterator(
            IDataItemLoader itemLoader,
            Func<IDataItemTransformation> transformationFactory 
            )
        {
            if (itemLoader == null)
            {
                throw new ArgumentNullException("itemLoader");
            }
            if (transformationFactory == null)
            {
                throw new ArgumentNullException("transformationFactory");
            }

            _itemLoader = itemLoader;
            _transformationFactory = transformationFactory;

            this.ResetIteration();

            var item0 = _itemLoader.Load(0);
            var transformed = _dataItemTransformation.Transform(item0);

            this.InputLength = transformed.InputLength;
        }

        public bool MoveNext()
        {
            var result = false;

            if (_currentIndex < _itemLoader.Count)
            {
                var item = _itemLoader.Load(_currentIndex);
                this.Current = _dataItemTransformation.Transform(item);

                _currentIndex++;

                result = true;
            }

            return
                result;
        }

        public void Reset()
        {
            this.ResetIteration();
        }

        public void Dispose()
        {
            //nothing to do
        }

        private void ResetIteration(
            )
        {
            _dataItemTransformation = _transformationFactory();
            _currentIndex = 0;
        }
    }
}