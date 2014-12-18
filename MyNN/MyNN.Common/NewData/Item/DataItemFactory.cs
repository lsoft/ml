using System;

namespace MyNN.Common.NewData.Item
{
    public class DataItemFactory : IDataItemFactory
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
                new DataItem(
                    input,
                    output);
        }
    }
}
