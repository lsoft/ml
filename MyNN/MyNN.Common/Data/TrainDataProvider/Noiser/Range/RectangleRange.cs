using System;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Data.TrainDataProvider.Noiser.Range
{
    /// <summary>
    /// Формирует маску по квадрату.
    /// </summary>
    [Serializable]
    public class RectangleRange : IRange
    {
        private readonly IRandomizer _randomizer;
        private readonly int _imageWidth;
        private readonly int _imageHeight;
        private readonly rint _rectWidth;
        private readonly rint _rectHeight;
        private readonly bool _needInvert;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="randomizer">Рандомайзер</param>
        /// <param name="imageWidth">Длина картинки в пикселях</param>
        /// <param name="imageHeight">Высота картинки в пикселях</param>
        /// <param name="rectWidth">Длина маски в пикселях (случ. величина)</param>
        /// <param name="rectHeight">Высота маски в пикселях (случ. величина)</param>
        /// <param name="needInvert">Необходимо ли инвертировать. false - шум применяется только ВНУТРИ квадрата, true - шум применяется только СНАРУЖИ квадрата.</param>
        public RectangleRange(
            IRandomizer randomizer,
            int imageWidth,
            int imageHeight,
            rint rectWidth,
            rint rectHeight,
            bool needInvert
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (rectWidth == null)
            {
                throw new ArgumentNullException("rectWidth");
            }
            if (rectHeight == null)
            {
                throw new ArgumentNullException("rectHeight");
            }
            if (rectWidth.Min < 0)
            {
                throw new ArgumentException("rectWidth.Min < 0");
            }
            if (rectWidth.Max > imageWidth)
            {
                throw new ArgumentException("rectWidth.Max > imageWidth");
            }
            if (rectHeight.Min < 0)
            {
                throw new ArgumentException("rectHeight.Min < 0");
            }
            if (rectHeight.Max > imageHeight)
            {
                throw new ArgumentException("rectHeight.Max > imageHeight");
            }

            _randomizer = randomizer;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
            _rectWidth = rectWidth;
            _rectHeight = rectHeight;
            _needInvert = needInvert;
        }

        public bool[] GetIndexMask(int dataLength)
        {
            if (dataLength != _imageWidth*_imageHeight)
            {
                throw new ArgumentException("dataLength != _imageWidth*_imageHeight");
            }

            var result = new bool[dataLength];

            var rectWidth = _rectWidth.Sample();
            var rectHeight = _rectHeight.Sample();

            var rectLeft = _randomizer.Next(_imageWidth - rectWidth);
            var rectTop = _randomizer.Next(_imageHeight - rectHeight);

            var diapWidth = new diapint(rectLeft, rectLeft + rectWidth - 1); //due to inclusive test
            var diapHeight = new diapint(rectTop, rectTop + rectHeight - 1); //due to inclusive test

            var index = 0;
            for (var row = 0; row < _imageHeight; row++)
            {
                for (var col = 0; col < _imageWidth; col++)
                {
                    var yo = diapHeight.IsValueInclusive(row) && diapWidth.IsValueInclusive(col);

                    if (_needInvert)
                    {
                        yo = !yo;
                    }

                    result[index] = yo;

                    index++;
                }
            }

            return result;
        }
    }
}