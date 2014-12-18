using System;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNN.Common.NewData.Noiser
{
    public class NoiserVisualizer
    {
        private readonly IRandomizer _randomizer;
        private readonly INoiser _noiser;

        public NoiserVisualizer(
            IRandomizer randomizer,
            INoiser noiser
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }

            _randomizer = randomizer;
            _noiser = noiser;
        }

        public void Visuzalize(
            IDataSet data,
            int width,
            int height,
            int scaleFactor
            )
        {
            var shuffler = new ArrayShuffler<IDataItem>(_randomizer, data);
            foreach (var di in shuffler)
            {
                var origBytes = di.Input.ConvertAll<float, byte>((a) => (byte)(a * 255));

                var noised = _noiser.ApplyNoise(di.Input);
                var noisedBytes = noised.ConvertAll<float, byte>((a) => (byte)(a * 255));

                unsafe
                {
                    fixed(byte* ob = origBytes)
                    fixed (byte* nb = noisedBytes)
                    {
                        using (var orig = IplImage.FromPixelData(width, height, 1, new IntPtr(ob)))
                        using (var gray = IplImage.FromPixelData(width, height, 1, new IntPtr(nb)))
                        using (var combined = new IplImage(width * 2 + 1, height, BitDepth.U8, 1))
                        using (var scaled = new IplImage(combined.Width * scaleFactor, combined.Height * scaleFactor, combined.Depth, 1))
                        {
                            combined.Zero();

                            combined.SetROI(
                                new CvRect(
                                    0, 
                                    0, 
                                    width, 
                                    height));
                            orig.Copy(combined);

                            combined.ResetROI();
                            combined.SetROI(
                                new CvRect(
                                    width + 1, 
                                    0, 
                                    width, 
                                    height));
                            gray.Copy(combined);
                            combined.ResetROI();

                            combined.Resize(scaled);

                            using (var w2 = new CvWindow("b"))
                            {
                                w2.ShowImage(scaled);
                                Cv.WaitKey();
                            }
                        }
                    }
                }

            }

        }
    }
}
