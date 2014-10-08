using System;
using MyNN.Common.ArtifactContainer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Structure;

namespace MyNN.MLP.MLPContainer
{
    public class MLPContainerHelper : IMLPContainerHelper
    {
        public MLPContainerHelper(
            )
        {
        }

        public IMLP Load(
            IArtifactContainer artifactContainer,
            string mlpName)
        {
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }

            var result = artifactContainer.LoadSerialized<Structure.MLP>(mlpName);

            //var result = _serializationHelper.LoadFromFile<MyNN.MLP2.Structure.MLP>(path);

            return result;
        }


        public void SaveMLP(
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IAccuracyRecord accuracyRecord)
        {
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }

            //сохраняем сеть
            artifactContainer.SaveSerialized(mlp, mlp.Name);
            
            //сохраняем файл с результатами
            artifactContainer.SaveSerialized(accuracyRecord.GetTextResults(), "_result.txt");

            ////сохраняем сеть
            //var mlpFilePath = Path.Combine(_rootFolder, mlp.Name);
            //_serializationHelper.SaveToFile(mlp, mlpFilePath);

            ////сохраняем файл с результатами
            //var txtFilePath = Path.Combine(_rootFolder, "_result.txt");
            //File.WriteAllText(txtFilePath, accuracyRecord.GetTextResults());
        }

    }
}