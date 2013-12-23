using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.BoltzmannMachines.LinearNReLU;
using MyNN.Data.TypicalDataProvider;
using MyNN.MLP2.Randomizer;

namespace MyNNConsoleApp
{
    public class TrainNReLUDBN
    {
        public void Train()
        {
            //обучение Linear - NReLU DBN
            var rndSeed = 1923443;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                "mnist/trainingset/",
                int.MaxValue);
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "mnist/testset/",
                int.MaxValue);
            validationData.GNormalize();

            var dbn = new DeepBeliefNetwork(
                randomizer,
                28, 28,
                784, 500);//, 2000, 50);

            dbn.Train(
                null,
                trainData,
                validationData,
                1,
                0.00002f,
                3,//20,
                30);

            //var bmp = new Bitmap(28*2 + 1, 28);
            //CreateContrastEnhancedBitmapFrom(bmp, 0, 0, trainData[0].Input);
            //CreateContrastEnhancedBitmapFrom(bmp, 29, 0, d.ToArray());
            //bmp.Save("_r.bmp");
        }
    }
}
