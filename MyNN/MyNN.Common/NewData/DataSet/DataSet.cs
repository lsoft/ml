using System;
using System.Collections;
using System.Collections.Generic;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet
{
    [Serializable]
    public class DataSet : IDataSet
    {
        private readonly IDataIteratorFactory _iteratorFactory;
        private readonly IDataItemLoader _itemLoader;
        private readonly IDataItemTransformationFactory _itemTransformationFactory;
        private readonly int _epochNumber;

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

        public int OutputLength
        {
            get;
            private set;
        }

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    _itemTransformationFactory.IsAutoencoderDataSet;
            }
        }

        public DataSet(
            IDataIteratorFactory iteratorFactory,
            IDataItemTransformationFactory itemTransformationFactory,
            IDataItemLoader itemLoader,
            int epochNumber
            )
        {
            if (iteratorFactory == null)
            {
                throw new ArgumentNullException("iteratorFactory");
            }
            if (itemTransformationFactory == null)
            {
                throw new ArgumentNullException("itemTransformationFactory");
            }
            if (itemLoader == null)
            {
                throw new ArgumentNullException("itemLoader");
            }

            _iteratorFactory = iteratorFactory;
            _itemTransformationFactory = itemTransformationFactory;
            _itemLoader = itemLoader;
            _epochNumber = epochNumber;

            var item0 = _itemLoader.Load(0);
            var transformed = itemTransformationFactory.CreateTransformation(0).Transform(item0);

            this.InputLength = transformed.InputLength;
            this.OutputLength = transformed.OutputLength;
        }

        public IDataIterator StartIterate(
            )
        {
            return 
                _iteratorFactory.CreateIterator(
                    _itemLoader,
                    () =>  _itemTransformationFactory.CreateTransformation(_epochNumber)
                    );
        }

        public IEnumerator<IDataItem> GetEnumerator()
        {
            return
                this.StartIterate();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}