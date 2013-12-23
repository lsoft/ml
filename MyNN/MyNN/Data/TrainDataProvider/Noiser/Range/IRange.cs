using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.Data.TrainDataProvider.Noiser.Range
{
    public interface IRange
    {
        void GetIndexes(
            int length,
            out int minIncludeIndex,
            out int maxExcludeIndex);

    }
}
