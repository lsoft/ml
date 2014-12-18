using System;
using System.Linq;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    [Serializable]
    public class ListDataItemTransformation : IDataItemTransformation
    {
        private readonly IDataItemTransformation[] _transformations;

        public bool IsAutoencoderDataSet
        {
            get
            {
                //если один трасформатор сделал итем автоекодерным, значит
                //output информация безвозвратно потеряна
                //таким образом, в списке достаточно только одного такого трансформатора

                return
                    _transformations.Any(j => j.IsAutoencoderDataSet);
            }
        }


        public ListDataItemTransformation(
            params IDataItemTransformation[] transformations
            )
        {
            if (transformations == null)
            {
                throw new ArgumentNullException("transformations");
            }
            if (transformations.Length == 0)
            {
                throw new ArgumentException("transformations.Length == 0");
            }

            _transformations = transformations;
        }

        public IDataItem Transform(IDataItem before)
        {
            if (before == null)
            {
                throw new ArgumentNullException("before");
            }

            var result = before;

            foreach (var t in _transformations)
            {
                result = t.Transform(result);
            }

            return
                result;
        }
    }
}