typedef struct
{
    float Sum;
    float C;
} KahanAccumulator;

inline KahanAccumulator GetKahanAcc(
    float startValue
    )
{
    KahanAccumulator result;
    result.Sum = startValue;
    result.C = 0.0;

    return result;
}

inline KahanAccumulator GetEmptyKahanAcc(
    )
{
    KahanAccumulator result;
    result.Sum = 0.0;
    result.C = 0.0;

    return result;
}

inline void KahanAddElement(
    KahanAccumulator* acc,
    float dataItem
    )
{
    float y = dataItem - acc->C;
    float t = acc->Sum + y;
    acc->C = (t - acc->Sum) - y;
    acc->Sum = t;
}

inline float KahanSum(
    float* data,
    int dataLength
    )
{
    if (dataLength == 0)
    {
        return 0.0;
    }

    float sum = data[0];
    float c = 0.0;
    for (int i = 1; i < dataLength; i++)
    {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
