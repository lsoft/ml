namespace MyNN.MLP2.Saver
{
    public interface IAccuracyRecord
    {
        /// <summary>
        /// Получить результаты в текстовом виде
        /// </summary>
        /// <returns></returns>
        string GetTextResults();
    }
}