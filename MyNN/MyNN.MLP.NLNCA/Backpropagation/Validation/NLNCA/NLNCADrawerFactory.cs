using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;

namespace MyNN.MLP.NLNCA.Backpropagation.Validation.NLNCA
{
    public class NLNCADrawerFactory : IDrawerFactory
    {
        private readonly IDataSet _validationData;
        private readonly IArtifactContainer _artifactContainer;
        private readonly IColorProvider _colorProvider;

        public NLNCADrawerFactory(
            IDataSet validationData,
            IArtifactContainer artifactContainer,
            IColorProvider colorProvider
            )
        {
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (artifactContainer == null)
            {
                throw new ArgumentNullException("artifactContainer");
            }
            if (colorProvider == null)
            {
                throw new ArgumentNullException("colorProvider");
            }
            _validationData = validationData;
            _artifactContainer = artifactContainer;
            _colorProvider = colorProvider;
        }

        public IDrawer CreateDrawer(
            IArtifactContainer containerForSave,
            int? epocheNumber
            )
        {
            if (containerForSave == null)
            {
                throw new ArgumentNullException("containerForSave");
            }

            return
                new NLNCADrawer(
                    _validationData,
                    _artifactContainer,
                    _colorProvider,
                    epocheNumber
                    );
        }
    }
}