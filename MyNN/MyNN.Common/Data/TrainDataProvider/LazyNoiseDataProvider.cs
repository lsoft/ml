using System;
using System.Collections.Generic;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.TrainDataProvider
{
    public class LazyNoiseDataProvider : ITrainDataProvider
    {
        private readonly IDataSet _trainData;
        private readonly INoiser _noiser;
        private readonly ISerializationHelper _serializationHelper;
        private readonly Func<int, INoiser> _noiserProvider;

        public LazyNoiseDataProvider(
            IDataSet trainData,
            INoiser noiser,
            ISerializationHelper serializationHelper
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
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }

            _trainData = trainData;
            _noiser = noiser;
            _serializationHelper = serializationHelper;
            _noiserProvider = null;
        }

        public LazyNoiseDataProvider(
            IDataSet trainData,
            Func<int, INoiser> noiserProvider)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (noiserProvider == null)
            {
                throw new ArgumentNullException("noiserProvider");
            }

            _trainData = trainData;
            _noiser = null;
            _noiserProvider = noiserProvider;
        }

        public IDataSet GetDataSet(int epocheNumber)
        {
            var result = new List<IDataItem>();

            foreach (var d in this._trainData)
            {
                var noiser = _noiser ?? _noiserProvider(epocheNumber);

                var di = new LazyDataItem(
                    d,
                    _serializationHelper.DeepClone(noiser),
                    _serializationHelper
                    );

                result.Add(di);
            }

            return
                new DataSet(result);
        }
    }
}