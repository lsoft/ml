using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Data.DataSetConverter
{
    public class  ListDataSetConverter : IDataSetConverter
    {
        private readonly IDataSetConverter[] _dataSetConverters;

        public ListDataSetConverter(
            params IDataSetConverter[] dataSetConverters)
        {
            if (dataSetConverters == null)
            {
                throw new ArgumentNullException("dataSetConverters");
            }

            _dataSetConverters = dataSetConverters;
        }

        public IDataSet Convert(IDataSet beforeTransformation)
        {
            if (beforeTransformation == null)
            {
                throw new ArgumentNullException("beforeTransformation");
            }

            var tds = beforeTransformation;
            foreach (var c in _dataSetConverters)
            {
                tds = c.Convert(tds);
            }

            return tds;
        }
    }
}
