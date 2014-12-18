using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;

namespace MyNN.Common.NewData.DataSet.ItemLoader
{
    public class FromArrayDataItemLoader : IDataItemLoader
    {
        private readonly INormalizer _normalizer;
        private readonly List<IDataItem> _list;

        public int Count
        {
            get
            {
                return
                    _list.Count;
            }
        }

        public FromArrayDataItemLoader(
            IEnumerable<IDataItem> list,
            INormalizer normalizer
            )
        {
            if (list == null)
            {
                throw new ArgumentNullException("list");
            }
            if (normalizer == null)
            {
                throw new ArgumentNullException("normalizer");
            }

            _normalizer = normalizer;
            _list = list.ToList();
        }

        public void Normalize(float bias = 0f)
        {
            foreach (var di in this._list)
            {
                _normalizer.Normalize(di.Input, bias);
            }
        }

        public void GNormalize()
        {
            foreach (var di in this._list)
            {
                _normalizer.GNormalize(di.Input);
            }
        }

        public IDataItem Load(int index)
        {
            return
                _list[index];
        }
    }
}