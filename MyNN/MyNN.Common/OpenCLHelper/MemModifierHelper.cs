using System;

namespace MyNN.Common.OpenCLHelper
{
    public class MemModifierHelper
    {
        public static string GetModifierSuffix(MemModifierEnum mme)
        {
            switch (mme)
            {
                case MemModifierEnum.NotSpecified:
                    return string.Empty;
                case MemModifierEnum.Local:
                    return "__local";
                case MemModifierEnum.Global:
                    return "__global";
                default:
                    throw new ArgumentOutOfRangeException("mme");
            }
        }
    }
}