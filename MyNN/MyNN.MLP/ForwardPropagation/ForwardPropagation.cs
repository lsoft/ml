using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation
{
    /// <summary>
    /// MLP Forward propagation
    /// </summary>
    public class ForwardPropagation : IForwardPropagation
    {
        private readonly IMLP _mlp;

        public IMLP MLP
        {
            get
            {
                return _mlp;
            }
        }

        private readonly ILayerContainer[] _mems;
        private readonly ILayerPropagator[] _propagators;

        private ILayerContainer _lastContainer
        {
            get
            {
                return
                    _mems.Last();
            }
        }

        public ForwardPropagation(
            ILayerContainer[] containers,
            ILayerPropagator[] propagators,
            IMLP mlp
            )
        {
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            #region validate

            if (containers == null)
            {
                throw new ArgumentNullException("containers");
            }
            if (propagators == null)
            {
                throw new ArgumentNullException("propagators");
            }
            if (containers.Length != mlp.Layers.Length)
            {
                throw new ArgumentException("containers.Length != mlp.Layers.Length");
            }
            if (propagators.Length != mlp.Layers.Length)
            {
                throw new ArgumentException("propagators.Length != mlp.Layers.Length");
            }
            if (containers.Any(j => j == null))
            {
                throw new ArgumentException("containers.Any(j => j == null)");
            }
            if (propagators[0] != null)
            {
                throw new ArgumentException("propagators[0] != null");
            }
            if (propagators.Skip(1).Any(j => j == null))
            {
                throw new ArgumentException("propagators.Skip(1).Any(j => j == null)");
            }

            #endregion

            _mems = containers;
            _propagators = propagators;
            _mlp = mlp;
        }

        public List<ILayerState> ComputeOutput(IDataSet dataSet)
        {
            TimeSpan propagationTime;
            var result = ComputeOutput(
                dataSet,
                out propagationTime);

            return result;
        }

        public List<ILayerState> ComputeOutput(IDataSet dataSet, out TimeSpan propagationTime)
        {
            var result = new List<ILayerState>();

            this.PushWeights();

            this.ClearAndPushHiddenLayers();

            var before = DateTime.Now;

            foreach (var d in dataSet)
            {
                this.Propagate(d);
                
                this.PopLastLayerState();

                var ls = _lastContainer.GetLayerState();

                result.Add(ls);
            }

            var after = DateTime.Now;
            propagationTime = (after - before);

            return result;
        }

        public List<IMLPState> ComputeState(IDataSet dataSet)
        {
            var result = new List<IMLPState>();

            this.PushWeights();

            this.ClearAndPushHiddenLayers();

            foreach (var d in dataSet)
            {
                this.Propagate(d);

                this.PopState();

                var listls = new List<ILayerState>();

                for (var layerIndex = 0; layerIndex < _mlp.Layers.Count(); layerIndex++)
                {
                    var ls = _mems[layerIndex].GetLayerState();
                    listls.Add(ls);
                }

                result.Add(
                    new MLPState(
                        listls.ToArray()));
            }

            return result;
        }

        public void ClearAndPushHiddenLayers()
        {
            for (var layerIndex = 1; layerIndex < _mlp.Layers.Length; layerIndex++)
            {
                _mems[layerIndex].ClearAndPushHiddenLayers();
            }
        }

        public void Propagate(IDataItem d)
        {
            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            //записываем значения в сеть
            _mems[0].PushInput(d.Input);
            
            //начинаем считать
            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                _propagators[layerIndex].ComputeLayer();
            }

            _propagators.Last().WaitForCalculationFinished();
        }

        public void PushWeights()
        {
            var layerCount = _mlp.Layers.Length;

            //веса оставшихся слоев
            for (var layerIndex = 1; layerIndex < layerCount; ++layerIndex)
            {
                var layer = _mlp.Layers[layerIndex];
                _mems[layerIndex].PushWeights(layer);
            }
        }

        public void PopState()
        {
            this.PopHiddenState();

            this.PopLastLayerState();
        }

        private void PopHiddenState()
        {
            var layerCount = _mlp.Layers.Length;

            //пишем результат обратно в сеть
            for (var layerIndex = 1; layerIndex < layerCount - 1; layerIndex++)
            {
                //читаем его из opencl
                _mems[layerIndex].PopHiddenState();
            }
        }

        private void PopLastLayerState()
        {
            //извлекаем из Opencl последний слой
            _lastContainer.PopLastLayerState();
        }

    }
}