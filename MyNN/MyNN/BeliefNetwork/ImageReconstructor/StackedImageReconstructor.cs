using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using MyNN.BoltzmannMachines;
using MyNN.Data;

namespace MyNN.BeliefNetwork.ImageReconstructor
{
    public class StackedImageReconstructor : IStackedImageReconstructor
    {
        private readonly IImageReconstructor _isolatedImageReconstructor;
        private readonly List<Func<float[], float[]>> _converterList;

        public StackedImageReconstructor(
            IImageReconstructor isolatedImageReconstructor)
        {
            if (isolatedImageReconstructor == null)
            {
                throw new ArgumentNullException("isolatedImageReconstructor");
            }

            _isolatedImageReconstructor = isolatedImageReconstructor;

            _converterList = new List<Func<float[], float[]>>();
        }

        public void AddConverter(
            Func<float[], float[]> converter)
        {
            if (converter == null)
            {
                throw new ArgumentNullException("converter");
            }

            _converterList.Add(converter);
        }


        public void AddPair(
            int dataItemIndexIntoDataSet, 
            float[] reconstructedData)
        {
            if (reconstructedData == null)
            {
                throw new ArgumentNullException("reconstructedData");
            }

            var d = reconstructedData;
            for (var ci = _converterList.Count - 1; ci >= 0; ci--)
            {
                var c = _converterList[ci];

                var cded = c(d);

                d = cded;
            }

            _isolatedImageReconstructor.AddPair(
                dataItemIndexIntoDataSet,
                d);
        }

        public Bitmap GetReconstructedBitmap()
        {
            return
                _isolatedImageReconstructor.GetReconstructedBitmap();
        }

        public int GetReconstructedImageCount()
        {
            return
                _isolatedImageReconstructor.GetReconstructedImageCount();
        }

    }
}
