using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Saver
{
    public interface IMLPSaver
    {
        void Save(
            string epocheRoot,
            IAccuracyRecord accuracyRecord,
            IMLP mlp);
    }
}