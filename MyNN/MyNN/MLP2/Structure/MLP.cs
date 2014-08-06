﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.Linq;
using Accord.Math;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure
{
    [Serializable]
    public class MLP : IMLP
    {
        public string Name
        {
            get;
            private set;
        }

        private readonly ILayerFactory _layerFactory;
        private volatile ILayer[] _layers;
        public ILayer[] Layers
        {
            get
            {
                return
                    _layers;
            }
        }

        public MLP(
            string name,
            ILayerFactory layerFactory,
            ILayer[] layerList)
        {
            if (name == null)
            {
                throw new ArgumentNullException("name");
            }
            if (layerFactory == null)
            {
                throw new ArgumentNullException("layerFactory");
            }
            if (layerList == null)
            {
                throw new ArgumentNullException("layerList");
            }
            if (layerList.Length < 2)
            {
                throw new ArgumentException("layerList.Length < 2");
            }
            if (layerList.Any(j => j == null))
            {
                throw new ArgumentException("layerList.Any(j => j == null)");
            }

            Name = name;
            _layerFactory = layerFactory;
            
            //формируем слои
            this._layers = layerList;
        }

        public string GetLayerInformation()
        {
            return
                string.Join(" -> ", this.Layers.ToList().ConvertAll(j => j.GetLayerInformation()));
        }

        public void OverwriteName(string newName)
        {
            if (newName == null)
            {
                throw new ArgumentNullException("newName");
            }

            this.Name = newName;
        }

        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с узкого слоя и до конца
        /// </summary>
        public void AutoencoderCutTail()
        {
            var lls = new ILayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this._layers = lls;

            //у последнего слоя убираем Bias нейрон
            var nll = this.Layers.Last();
            nll.RemoveBiasNeuron();
        }

        /// <summary>
        /// Убрать последний слой
        /// </summary>
        public void CutLastLayer()
        {
            var lls = new ILayer[this.Layers.Length - 1];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this._layers = lls;
        }


        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с первого и до узкого слоя
        /// </summary>
        public void AutoencoderCutHead()
        {
            var lls = new ILayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, this.Layers.Length - lls.Length, lls, 0, lls.Length);

            this._layers = lls;
        }

        public void AddLayer(
            IFunction activationFunction,
            int nonBiasNeuronCount,
            bool isNeedBiasNeuron)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            var lastl = this._layers[this._layers.Length - 1];
            if (!lastl.IsBiasNeuronExists)
            {
                lastl.AddBiasNeuron();
            }

            var newl = new ILayer[this._layers.Length + 1];
            this._layers.CopyTo(newl, 0);

            var bornLayer = _layerFactory.CreateLayer(
                activationFunction,
                nonBiasNeuronCount,
                lastl.NonBiasNeuronCount,
                isNeedBiasNeuron,
                true
                );
            newl[this._layers.Length] = bornLayer;

            this._layers = newl;
        }

        public IMLPConfiguration GetConfiguration()
        {
            return 
                new MLPConfiguration(
                    this.Layers.ConvertAll(j => j.GetConfiguration()));
        }
    }
}
