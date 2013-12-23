using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace MyNN.MLP2.Randomizer
{
    public interface IRandomizer
    {
        int Next(int maxValue);
        
        float Next();

        void NextBytes(byte[] buffer);
    }
}
