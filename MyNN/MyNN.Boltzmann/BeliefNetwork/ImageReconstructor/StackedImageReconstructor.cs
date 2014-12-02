using System;
using System.Collections.Generic;
using System.Drawing;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter;
using MyNN.Boltzmann.BoltzmannMachines;

namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor
{
    public class StackedImageReconstructor : IStackedImageReconstructor
    {
        private readonly IImageReconstructor _isolatedImageReconstructor;
        private readonly List<IDataArrayConverter> _converterList;

        public StackedImageReconstructor(
            IImageReconstructor isolatedImageReconstructor)
        {
            if (isolatedImageReconstructor == null)
            {
                throw new ArgumentNullException("isolatedImageReconstructor");
            }

            _isolatedImageReconstructor = isolatedImageReconstructor;

            _converterList = new List<IDataArrayConverter>();
        }

        public void AddConverter(
            IDataArrayConverter converter
            )
        {
            if (converter == null)
            {
                throw new ArgumentNullException("converter");
            }

            _converterList.Add(converter);
        }

        public Bitmap GetReconstructedBitmap(
            int startDataItemIndexIntoDataSet,
            List<float[]> reconstructedDataList
            )
        {
            if (reconstructedDataList == null)
            {
                throw new ArgumentNullException("reconstructedDataList");
            }

            var processed = reconstructedDataList;
            for (var ci = _converterList.Count - 1; ci >= 0; ci--)
            {
                var converter = _converterList[ci];

                var forOneIteration = new List<float[]>();
                foreach (var di in processed)
                {
                    var dip = converter.Convert(di);
                    forOneIteration.Add(dip);
                }

                processed = forOneIteration;
            }


            return
                _isolatedImageReconstructor.GetReconstructedBitmap(
                    startDataItemIndexIntoDataSet,
                    processed
                    );
        }

        public int GetReconstructedImageCount()
        {
            return
                _isolatedImageReconstructor.GetReconstructedImageCount();
        }

    }
}
