using System.Text;
using System.Threading.Tasks;

namespace MyNN.Data.DataSetConverter
{
    public interface IDataSetConverter
    {
        IDataSet Convert(IDataSet beforeTransformation);
    }
}
