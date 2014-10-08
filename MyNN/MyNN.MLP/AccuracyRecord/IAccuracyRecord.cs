namespace MyNN.MLP.AccuracyRecord
{
    public interface IAccuracyRecord
    {
        /// <summary>
        /// �������� ������� ������ ������ �����
        /// </summary>
        float PerItemError
        {
            get;
        }

        /// <summary>
        /// �������� ���������� � ��������� ����
        /// </summary>
        /// <returns></returns>
        string GetTextResults();

        /// <summary>
        /// ���������� ��� ���������� ��������� � ����������, ����� �����
        /// </summary>
        /// <param name="accuracyRecord"></param>
        bool IsBetterThan(IAccuracyRecord accuracyRecord);
    }
}