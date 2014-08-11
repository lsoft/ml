typedef struct
{
    float4 Sum;
    float4 C;
} KahanAccumulator4;

inline float ReduceAcc4(
    KahanAccumulator4* acc4
    )
{
    KahanAccumulator acc = GetEmptyKahanAcc();

    KahanAddElement(&acc, acc4->Sum.s0);
    KahanAddElement(&acc, acc4->Sum.s1);
    KahanAddElement(&acc, acc4->Sum.s2);
    KahanAddElement(&acc, acc4->Sum.s3);

    return 
        acc.Sum;
}

inline KahanAccumulator4 GetEmptyKahanAcc4(
    )
{
    KahanAccumulator4 result;
    result.Sum = 0.0;
    result.C = 0.0;

    return result;
}

inline void KahanAddElement4(
    KahanAccumulator4* acc,
    float4 dataItem
    )
{
    float4 y = dataItem - acc->C;
    float4 t = acc->Sum + y;
    acc->C = (t - acc->Sum) - y;
    acc->Sum = t;
}

inline float4 KahanSum4(
    float4* data,
    int dataLength
    )
{
    if (dataLength == 0)
    {
        return 0.0;
    }

    float4 sum = data[0];
    float4 c = 0.0;
    for (int i = 1; i < dataLength; i++)
    {
        float4 y = data[i] - c;
        float4 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
