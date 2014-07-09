using System;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Saver
{
    public class ConsoleMLPSaver : IMLPSaver
    {
        public void Save(
            string epocheRoot,
            IAccuracyRecord accuracyRecord,
            IMLP mlp)
        {
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            ConsoleAmbientContext.Console.WriteLine(
                "[ConsoleMLPSaver] MLP {0} must be saved {1}, validation per-item error = {2}, correct {3} out of {4}",
                mlp.GetLayerInformation(),
                epocheRoot,
                accuracyRecord.ValidationPerItemError,
                accuracyRecord.CorrectCount,
                accuracyRecord.ToString()
                );
        }
    }
}