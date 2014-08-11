typedef struct
{
    float16 Sum;
    float16 C;
} KahanAccumulator16;

inline float ReduceAcc16(
    KahanAccumulator16* acc16
    )
{
    KahanAccumulator acc = GetEmptyKahanAcc();

    KahanAddElement(&acc, acc16->Sum.s0);
    KahanAddElement(&acc, acc16->Sum.s1);
    KahanAddElement(&acc, acc16->Sum.s2);
    KahanAddElement(&acc, acc16->Sum.s3);
    
    KahanAddElement(&acc, acc16->Sum.s4);
    KahanAddElement(&acc, acc16->Sum.s5);
    KahanAddElement(&acc, acc16->Sum.s6);
    KahanAddElement(&acc, acc16->Sum.s7);
    
    KahanAddElement(&acc, acc16->Sum.s8);
    KahanAddElement(&acc, acc16->Sum.s9);
    KahanAddElement(&acc, acc16->Sum.sa);
    KahanAddElement(&acc, acc16->Sum.sb);
    
    KahanAddElement(&acc, acc16->Sum.sc);
    KahanAddElement(&acc, acc16->Sum.sd);
    KahanAddElement(&acc, acc16->Sum.se);
    KahanAddElement(&acc, acc16->Sum.sf);

    return 
        acc.Sum;
}

inline KahanAccumulator16 GetEmptyKahanAcc16(
    )
{
    KahanAccumulator16 result;
    result.Sum = 0.0;
    result.C = 0.0;

    return result;
}

inline void KahanAddElement16(
    KahanAccumulator16* acc,
    float16 dataItem
    )
{
    float16 y = dataItem - acc->C;
    float16 t = acc->Sum + y;
    acc->C = (t - acc->Sum) - y;
    acc->Sum = t;
}

inline float16 KahanSum16(
    float16* data,
    int dataLength
    )
{
    if (dataLength == 0)
    {
        return 0.0;
    }

    float16 sum = data[0];
    float16 c = 0.0;
    for (int i = 1; i < dataLength; i++)
    {
        float16 y = data[i] - c;
        float16 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
