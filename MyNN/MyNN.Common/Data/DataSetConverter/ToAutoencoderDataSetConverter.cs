using System;

namespace MyNN.Common.Data.DataSetConverter
{
    public class ToAutoencoderDataSetConverter : IDataSetConverter
    {
        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var result =
                new DataSet(
                    beforeTransformation.Data.ConvertAll(j => (IDataItem)new DenseDataItem(j.Input, j.Input))
                    );

            return result;
        }
    }
}