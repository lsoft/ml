using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Data;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two
{
    /// <summary>
    /// MLP Forward propagation implemented in CPU-oriented (Intel) OpenCL
    /// </summary>
    public class CPUForwardPropagation2 : IForwardPropagation
    {
        private readonly IMLP _mlp;

        public IMLP MLP
        {
            get
            {
                return _mlp;
            }
        }

        private readonly CLProvider _clProvider;

        private readonly ILayerMemContainer[] _mems;
        private readonly ICPULayerPropagator[] _propagators;

        private ILayerMemContainer _lastContainer
        {
            get
            {
                return
                    _mems.Last();
            }
        }

        public CPUForwardPropagation2(
            ILayerMemContainer[] mems,
            ICPULayerPropagator[] propagators,
            IMLP mlp,
            CLProvider clProvider
            )
        {
            if (mems == null)
            {
                throw new ArgumentNullException("mems");
            }
            if (propagators == null)
            {
                throw new ArgumentNullException("propagators");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mems.Length != mlp.Layers.Length)
            {
                throw new ArgumentException("mems.Length != mlp.Layers.Length");
            }
            if (propagators.Length != mlp.Layers.Length)
            {
                throw new ArgumentException("propagators.Length != mlp.Layers.Length");
            }
            if (mems.Any(j => j == null))
            {
                throw new ArgumentException("mems.Any(j => j == null)");
            }
            if (propagators[0] != null)
            {
                throw new ArgumentException("propagators[0] != null");
            }
            if (propagators.Skip(1).Any(j => j == null))
            {
                throw new ArgumentException("propagators.Skip(1).Any(j => j == null)");
            }

            _mems = mems;
            _propagators = propagators;
            _mlp = mlp;
            _clProvider = clProvider;

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

        public void Propagate(DataItem d)
        {
            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            PushInput(d);
            
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            //начинаем считать
            var layerCount = _mlp.Layers.Length;

            for (var layerIndex = 1; layerIndex < layerCount; layerIndex++)
            {
                _propagators[layerIndex].ComputeLayer();
            }

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
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

        /// <summary>
        /// распаковывает значения из сети в массивы для opencl
        /// </summary>
        private void PushInput(DataItem d)
        {
            if (d == null)
            {
                throw new ArgumentNullException("d");
            }

            //записываем значения в сеть
            _mems[0].PushInput(d.Input);
        }

    }
}