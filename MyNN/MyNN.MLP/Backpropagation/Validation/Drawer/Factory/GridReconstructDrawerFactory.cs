using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Visualizer.Factory;

namespace MyNN.MLP.Backpropagation.Validation.Drawer.Factory
{
    public class GridReconstructDrawerFactory : IDrawerFactory
    {
        private readonly IVisualizerFactory _visualizerFactory;
        private readonly IDataSet _validationData;
        private readonly int _visualizeCount;

        public GridReconstructDrawerFactory(
            IVisualizerFactory visualizerFactory,
            IDataSet validationData,
            int visualizeCount
            )
        {
            if (visualizerFactory == null)
            {
                throw new ArgumentNullException("visualizerFactory");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }

            _visualizerFactory = visualizerFactory;
            _validationData = validationData;
            _visualizeCount = visualizeCount;
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
                new GridReconstructDrawer(
                    _visualizerFactory,
                    _validationData,
                    _visualizeCount,
                    containerForSave
                    );
        }
    }
}