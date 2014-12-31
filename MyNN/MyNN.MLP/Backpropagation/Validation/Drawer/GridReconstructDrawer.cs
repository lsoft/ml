using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Visualizer;
using MyNN.Common.NewData.Visualizer.Factory;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.Drawer
{
    public class GridReconstructDrawer : IDrawer
    {
        private readonly IVisualizerFactory _visualizerFactory;
        private readonly IDataSet _validationData;
        private readonly int _visualizeCount;
        private readonly IArtifactContainer _containerForSave;
        
        private int _startIndex;
        private int _currentIndex;
        private IDataIterator _validationDataIterator;
        private IVisualizer _visualizer;

        public GridReconstructDrawer(
            IVisualizerFactory visualizerFactory,
            IDataSet validationData,
            int visualizeCount,
            IArtifactContainer containerForSave
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
            if (containerForSave == null)
            {
                throw new ArgumentNullException("containerForSave");
            }

            _visualizerFactory = visualizerFactory;
            _validationData = validationData;
            _visualizeCount = visualizeCount;
            _containerForSave = containerForSave;

            _startIndex = -1;
        }

        public void SetSize(
            int netResultCount
            )
        {
            if (_startIndex != -1)
            {
                throw new InvalidOperationException("Установить размер можно только один раз");
            }

            _startIndex = (int)((DateTime.Now.Millisecond / 1000f) * (Math.Min(_validationData.Count, netResultCount) - _visualizeCount));
            _currentIndex = 0;
            _validationDataIterator = _validationData.StartIterate();
            _visualizer = _visualizerFactory.CreateVisualizer(
                Math.Min(
                    _visualizeCount,
                    netResultCount
            ));

            _validationDataIterator.MoveNext();
        }

        public void DrawItem(
            ILayerState netResult
            )
        {
            if (netResult == null)
            {
                throw new ArgumentNullException("netResult");
            }

            if (_validationData.IsAutoencoderDataSet)
            {
                if (_currentIndex < _visualizeCount)
                {
                    _visualizer.VisualizeGrid(
                        netResult.NState
                        );
                }

                if (_currentIndex >= _startIndex)
                {
                    if (_currentIndex < _startIndex + _visualizeCount)
                    {
                        _visualizer.VisualizePair(
                            new Pair<float[], float[]>(
                            _validationDataIterator.Current.Output,
                            netResult.NState)
                            );
                    }
                }

                //using (var s = _containerForSave.GetWriteStreamForResource("grid.bmp"))
                //{
                //    _visualizer.SaveAsGrid(
                //        s,
                //        netResult.ConvertAll(j => j.NState).Take(_visualizeAsGridCount).ToList());

                //    s.Flush();
                //}

                ////со случайного индекса
                //var startIndex = (int) ((DateTime.Now.Millisecond/1000f)*(Math.Min(_validationData.Count, netResult.Count) - _visualizeAsPairCount));

                //var pairList = new List<Pair<float[], float[]>>();
                //foreach (var pair in netResult.ZipEqualLength(_validationData).Skip(startIndex).Take(_visualizeAsPairCount))
                //{
                //    var netResult = pair.Value1;
                //    var testItem = pair.Value2;

                //    pairList.Add(
                //        new Pair<float[], float[]>(
                //            testItem.Output,
                //            netResult.NState));
                //}

                //using (var s = _containerForSave.GetWriteStreamForResource("reconstruct.bmp"))
                //{
                //    _visualizer.SaveAsPairList(
                //        s,
                //        pairList);

                //    s.Flush();
                //}

                _currentIndex++;
                _validationDataIterator.MoveNext();
            }
        }

        public void Save()
        {
            using (var s = _containerForSave.GetWriteStreamForResource("grid.bmp"))
            {
                _visualizer.SaveGrid(
                    s
                    );

                s.Flush();
            }

            using (var s = _containerForSave.GetWriteStreamForResource("reconstruct.bmp"))
            {
                _visualizer.SavePairs(
                    s
                    );

                s.Flush();
            }

            if (_validationDataIterator != null)
            {
                _validationDataIterator.Dispose();
            }

        }
    }
}
