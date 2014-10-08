using System;

namespace MyNN.Common.Data.DataSetConverter
{
    public class NoConvertDataSetConverter : IDataSetConverter
    {
        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            return
                beforeTransformation;
        }
    }
}