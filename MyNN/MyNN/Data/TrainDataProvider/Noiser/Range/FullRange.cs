namespace MyNN.Data.TrainDataProvider.Noiser.Range
{
    public class FullRange : IRange
    {
        public void GetIndexes(
            int length,
            out int minIncludeIndex,
            out int maxExcludeIndex)
        {
            minIncludeIndex = 0;
            maxExcludeIndex = length;
        }
    }
}