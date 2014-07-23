using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Container
{
    public interface IMLPContainer : IMLPReadContainer
    {
        void Save(
            IMLP mlp,
            IAccuracyRecord accuracyRecord
            );

        Stream GetWriteStreamForResource(string resourceName);

        IMLPContainer GetChildContainer(string containerName);
        
        void Clear();

        void DeleteResource(string resourceName);
    }
}
