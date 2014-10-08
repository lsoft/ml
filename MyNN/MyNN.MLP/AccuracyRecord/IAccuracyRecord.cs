namespace MyNN.MLP.AccuracyRecord
{
    public interface IAccuracyRecord
    {
        /// <summary>
        /// Получить среднюю ошибку одного итема
        /// </summary>
        float PerItemError
        {
            get;
        }

        /// <summary>
        /// Получить результаты в текстовом виде
        /// </summary>
        /// <returns></returns>
        string GetTextResults();

        /// <summary>
        /// Сравнивает два результата валидации и определяет, какой лучше
        /// </summary>
        /// <param name="accuracyRecord"></param>
        bool IsBetterThan(IAccuracyRecord accuracyRecord);
    }
}