using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Data.TrainDataProvider.Noiser;

namespace MyNN.Data.TrainDataProvider
{
    public class NoiseDataProvider : ITrainDataProvider
    {
        private readonly DataSet _trainData;
        private readonly INoiser _noiser;

        public bool IsAuencoderDataSet
        {
            get
            {
                return this._trainData.IsAuencoderDataSet;
            }
        }

        public bool IsClassificationAuencoderDataSet
        {
            get
            {
                return this._trainData.IsClassificationAuencoderDataSet;
            }
        }

        public NoiseDataProvider(
            DataSet trainData,
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
        }


        public DataSet GetDeformationDataSet(int epocheNumber)
        {
            var result = new List<DataItem>();

            foreach (var d in this._trainData)
            {
                var id = this._noiser.ApplyNoise(d.Input);

                var di = new DataItem(
                    id,
                    d.Output);

                result.Add(di);
            }

            return 
                new DataSet(
                    result,
                    this._trainData.Visualizer);
        }
    }
}
