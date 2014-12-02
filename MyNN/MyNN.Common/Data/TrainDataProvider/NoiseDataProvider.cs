using System;
using System.Collections.Generic;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.TrainDataProvider
{
    public class NoiseDataProvider : ITrainDataProvider
    {
        private readonly IDataSet _trainData;
        private readonly INoiser _noiser;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly Func<int, INoiser> _noiserProvider;

        public NoiseDataProvider(
            IDataSet trainData,
            INoiser noiser,
            IDataItemFactory dataItemFactory
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
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _trainData = trainData;
            _noiser = noiser;
            _dataItemFactory = dataItemFactory;
            _noiserProvider = null;
        }

        public NoiseDataProvider(
            IDataSet trainData,
            Func<int, INoiser> noiserProvider,
            IDataItemFactory dataItemFactory
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
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _trainData = trainData;
            _noiser = null;
            _noiserProvider = noiserProvider;
            _dataItemFactory = dataItemFactory;
        }

        public IDataSet GetDataSet(int epocheNumber)
        {
            var result = new List<IDataItem>();

            foreach (var d in this._trainData.Data)
            {
                var noiser = _noiser ?? _noiserProvider(epocheNumber);

                var noisedData =
                    noiser != null
                        ? noiser.ApplyNoise(d.Input)
                        : d.Input.CloneArray();

                var di = _dataItemFactory.CreateDataItem(
                    noisedData,
                    d.Output);

                result.Add(di);
            }

            return 
                new DataSet(result);
        }
    }
}
