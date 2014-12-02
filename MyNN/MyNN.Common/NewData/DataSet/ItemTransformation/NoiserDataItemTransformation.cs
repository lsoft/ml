using System;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    [Serializable]
    public class NoiserDataItemTransformation : IDataItemTransformation
    {
        private readonly IDataItemFactory _dataItemFactory;
        private readonly int _epochNumber;
        private readonly INoiser _noiser;
        private readonly Func<int, INoiser> _noiserProvider;

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    false;
            }
        }

        public NoiserDataItemTransformation(
            IDataItemFactory dataItemFactory,
            int epochNumber,
            INoiser noiser,
            Func<int, INoiser> noiserProvider
            )
        {
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (noiser == null && noiserProvider == null)
            {
                throw new ArgumentException("noiser == null && noiserProvider == null");
            }
            if (noiser != null && noiserProvider != null)
            {
                throw new ArgumentException("noiser != null && noiserProvider != null");
            }

            _dataItemFactory = dataItemFactory;
            _epochNumber = epochNumber;
            _noiser = noiser;
            _noiserProvider = noiserProvider;
        }

        public IDataItem Transform(IDataItem before)
        {
            if (before == null)
            {
                throw new ArgumentNullException("before");
            }

            var noiser = _noiser ?? _noiserProvider(_epochNumber);

            var noisedData =
                noiser != null
                    ? noiser.ApplyNoise(before.Input)
                    : before.Input.CloneArray();

            var newItem = _dataItemFactory.CreateDataItem(
                noisedData,
                before.Output);

            return
                newItem;
        }
    }
}