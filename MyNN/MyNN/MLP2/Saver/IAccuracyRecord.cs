namespace MyNN.MLP2.Saver
{
    public interface IAccuracyRecord
    {
        float ValidationPerItemError
        {
            get;
        }

        int TotalCount
        {
            get;
        }

        int CorrectCount
        {
            get;
        }

        float CorrectPercent
        {
            get;
        }
    }
}