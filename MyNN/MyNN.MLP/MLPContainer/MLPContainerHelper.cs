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
            artifactContainer.SaveString(
                accuracyRecord.GetTextResults(),
                "_result.txt"
                );
        }

    }
}