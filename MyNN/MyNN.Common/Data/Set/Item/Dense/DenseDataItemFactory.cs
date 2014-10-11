using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Common.Data.Set.Item.Dense
{
    public class DenseDataItemFactory : IDataItemFactory
    {
        public IDataItem CreateDataItem(
            float[] input,
            float[] output)
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
                new DenseDataItem(
                    input,
                    output);
        }
    }
}
