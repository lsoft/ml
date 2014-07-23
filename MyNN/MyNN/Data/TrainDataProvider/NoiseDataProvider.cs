﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data.TrainDataProvider.Noiser;

namespace MyNN.Data.TrainDataProvider
{
    public class NoiseDataProvider : ITrainDataProvider
    {
        private readonly IDataSet _trainData;
        private readonly INoiser _noiser;
        private readonly Func<int, INoiser> _noiserProvider;

        public bool IsAuencoderDataSet
        {
            get
            {
                return this._trainData.IsAuencoderDataSet;
            }
        }

        public NoiseDataProvider(
            IDataSet trainData,
            INoiser noiser)
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }

            _trainData = trainData;
            _noiser = noiser;
            _noiserProvider = null;
        }

        public NoiseDataProvider(
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
            var result = new List<DataItem>();

            foreach (var d in this._trainData)
            {
                var noiser = _noiser ?? _noiserProvider(epocheNumber);

                var id = noiser.ApplyNoise(d.Input);

                var di = new DataItem(
                    id,
                    d.Output);

                result.Add(di);
            }

            return 
                new DataSet(result);
        }
    }
}
