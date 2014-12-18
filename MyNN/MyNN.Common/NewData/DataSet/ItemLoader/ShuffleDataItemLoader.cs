using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.NewData.Item;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.DataSet.ItemLoader
{
    [Serializable]
    public class ShuffleDataItemLoader : IDataItemLoader
    {
        private readonly IRandomizer _randomizer;
        private readonly IDataItemLoader _itemLoader;

        private List<int> _indexList;

        public int Count
        {
            get
            {
                return
                    _itemLoader.Count;
            }
        }

        public ShuffleDataItemLoader(
            IRandomizer randomizer,
            IDataItemLoader itemLoader
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (itemLoader == null)
            {
                throw new ArgumentNullException("itemLoader");
            }

            _randomizer = randomizer;
            _itemLoader = itemLoader;

            this.Fill();
        }

        public IDataItem Load(int index)
        {
            var index2 = _indexList[index];

            return
                _itemLoader.Load(index2);
        }

        public void Normalize(float bias = 0f)
        {
            _itemLoader.Normalize(bias);
        }

        public void GNormalize()
        {
            _itemLoader.GNormalize();
        }

        private void Fill(
            )
        {
            _indexList = Enumerable.Range(0, _itemLoader.Count).ToList();
            for (var i = 0; i < _itemLoader.Count - 1; i++)
            {
                if (_randomizer.Next() >= 0.5f)
                {
                    var newIndex = _randomizer.Next(_indexList.Count);

                    var tmp = _indexList[i];
                    _indexList[i] = _indexList[newIndex];
                    _indexList[newIndex] = tmp;
                }
            }
        }
    }
}