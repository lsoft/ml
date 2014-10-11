using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Set.Item.Lazy
{
    public class LazyDataItemFactory : IDataItemFactory 
    {
        private readonly INoiser _noiser;
        private readonly ISerializationHelper _serializationHelper;

        public LazyDataItemFactory(
            INoiser noiser,
            ISerializationHelper serializationHelper
            )
        {
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }

            _noiser = noiser;
            _serializationHelper = serializationHelper;
        }

        public IDataItem CreateDataItem(
            float[] input, 
            float[] output
            )
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }
            if (output == null)
            {
                throw new ArgumentNullException("output");
            }

            return 
                new LazyDataItem(
                    input,
                    output,
                    _serializationHelper.DeepClone(_noiser),
                    _serializationHelper
                    );
        }

    }
}
