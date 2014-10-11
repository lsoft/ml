using System;
using System.Collections.Generic;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Lazy;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.TrainDataProvider
{
    public class LazyNoiseDataProvider : ITrainDataProvider
    {
        private readonly IDataSet _trainData;
        private readonly INoiser _noiser;
        private readonly Func<INoiser, IDataItemFactory> _dataItemFactoryFunc;
        private readonly Func<int, INoiser> _noiserProvider;

        public LazyNoiseDataProvider(
            IDataSet trainData,
            INoiser noiser,
            Func<INoiser, IDataItemFactory> dataItemFactoryFunc
            )
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }
            if (dataItemFactoryFunc == null)
            {
                throw new ArgumentNullException("dataItemFactoryFunc");
            }

            _trainData = trainData;
            _noiser = noiser;
            _dataItemFactoryFunc = dataItemFactoryFunc;
            _noiserProvider = null;
        }

        public LazyNoiseDataProvider(
            IDataSet trainData,
            Func<int, INoiser> noiserProvider,
            Func<INoiser, IDataItemFactory> dataItemFactoryFunc
            )
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (noiserProvider == null)
            {
                throw new ArgumentNullException("noiserProvider");
            }
            if (dataItemFactoryFunc == null)
            {
                throw new ArgumentNullException("dataItemFactoryFunc");
            }

            _trainData = trainData;
            _noiser = null;
            _noiserProvider = noiserProvider;
            _dataItemFactoryFunc = dataItemFactoryFunc;
        }

        public IDataSet GetDataSet(int epocheNumber)
        {
            var result = new List<IDataItem>();

            foreach (var d in this._trainData)
            {
                var noiser = _noiser ?? _noiserProvider(epocheNumber);

                var dataItemFactory = _dataItemFactoryFunc(noiser);

                var di = dataItemFactory.CreateDataItem(
                    d.Input,
                    d.Output
                    );

                result.Add(di);
            }

            return
                new DataSet(result);
        }
    }
}