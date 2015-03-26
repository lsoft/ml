using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.Tests.MLP2.Convolution
{
    [TestClass]
    public class CSharpConvolutionCalculatorFixture
    {
        [TestMethod]
        public void TestOverwrite()
        {
            const int datadim = 5;
            const int datatotal = datadim*datadim;

            const int kerneldim = 3;
            const int kerneltotal = kerneldim*kerneldim;

            const int targetdim = datadim - kerneldim + 1;
            const int targettotal = targetdim*targetdim;

            var datasd = new Dimension(2, datadim, datadim);
            var kernelsd = new Dimension(2, kerneldim, kerneldim);
            var targetsd = new Dimension(2, targetdim, targetdim);

            var data = new float[datatotal];

            var kernel = new float[kerneltotal];
            var bias = new float[1];

            var target = new float[targettotal];

            var d = new ReferencedSquareFloat(
                datasd,
                data,
                0
                );

            var k = new ReferencedKernelBiasContainer(
                kernelsd,
                kernel,
                0,
                bias,
                0
                );


            var t = new ReferencedSquareFloat(
                targetsd,
                target,
                0
                );

            //data:
            var datavalues = new float[,]
            {
                { 1,  2,  3,  2,  1, },
                { 3,  2, -1,  2,  3, },
                { 1,  1,  0,  1,  1, },
                { 0,  0, -2,  1,  1, },
                { 3,  0,  0,  2,  4, },
            };

            FillSquare(d, datavalues);


            //kernel
            var kernelvalues = new float[,]
            {
                { 1,  2,  3, },
                { 3, -2,  1, },
                { 1,  1,  0, },
            };
            bias[0] = 0;

            FillKb(k, kernelvalues);

            //right target:
            var righttargervalues = new float[,]
            {
                { 20,  25,   7, },
                {  5,   8,  10, },
                {  4,   9,   0, },
            };

            var cc = new NaiveConvolutionCalculator(
                );
            
            cc.CalculateConvolutionWithOverwrite(
                k,
                d,
                t
                );

            float maxDiff;
            var r = ArrayOperations.ValuesAreEqual(
                target,
                righttargervalues.ToLine(),
                float.Epsilon,
                out maxDiff
                );

            Assert.IsTrue(r);

        }

        [TestMethod]
        public void TestOverwriteBias()
        {
            const int datadim = 5;
            const int datatotal = datadim * datadim;

            const int kerneldim = 3;
            const int kerneltotal = kerneldim * kerneldim;

            const int targetdim = datadim - kerneldim + 1;
            const int targettotal = targetdim * targetdim;

            var datasd = new Dimension(2, datadim, datadim);
            var kernelsd = new Dimension(2, kerneldim, kerneldim);
            var targetsd = new Dimension(2, targetdim, targetdim);

            var data = new float[datatotal];

            var kernel = new float[kerneltotal];
            var bias = new float[1];

            var target = new float[targettotal];

            var d = new ReferencedSquareFloat(
                datasd,
                data,
                0
                );

            var k = new ReferencedKernelBiasContainer(
                kernelsd,
                kernel,
                0,
                bias,
                0
                );


            var t = new ReferencedSquareFloat(
                targetsd,
                target,
                0
                );

            //data:
            var datavalues = new float[,]
            {
                { 1,  2,  3,  2,  1, },
                { 3,  2, -1,  2,  3, },
                { 1,  1,  0,  1,  1, },
                { 0,  0, -2,  1,  1, },
                { 3,  0,  0,  2,  4, },
            };

            FillSquare(d, datavalues);


            //kernel
            var kernelvalues = new float[,]
            {
                { 1,  2,  3, },
                { 3, -2,  1, },
                { 1,  1,  0, },
            };
            bias[0] = -1;

            FillKb(k, kernelvalues);

            //right target:
            var righttargervalues = new float[,]
            {
                { 19,  24,   6, },
                {  4,   7,   9, },
                {  3,   8,   -1, },
            };

            var cc = new NaiveConvolutionCalculator(
                );

            cc.CalculateConvolutionWithOverwrite(
                k,
                d,
                t
                );

            float maxDiff;
            var r = ArrayOperations.ValuesAreEqual(
                target,
                righttargervalues.ToLine(),
                float.Epsilon,
                out maxDiff
                );

            Assert.IsTrue(r);

        }

        [TestMethod]
        public void TestIncrement()
        {
            const int datadim = 5;
            const int datatotal = datadim * datadim;

            const int kerneldim = 3;
            const int kerneltotal = kerneldim * kerneldim;

            const int targetdim = datadim - kerneldim + 1;
            const int targettotal = targetdim * targetdim;

            var datasd = new Dimension(2, datadim, datadim);
            var kernelsd = new Dimension(2, kerneldim, kerneldim);
            var targetsd = new Dimension(2, targetdim, targetdim);

            var data = new float[datatotal];

            var kernel = new float[kerneltotal];
            var bias = new float[1];

            var target = new float[targettotal];

            var d = new ReferencedSquareFloat(
                datasd,
                data,
                0
                );

            var k = new ReferencedKernelBiasContainer(
                kernelsd,
                kernel,
                0,
                bias,
                0
                );


            var t = new ReferencedSquareFloat(
                targetsd,
                target,
                0
                );

            //data:
            var datavalues = new float[,]
            {
                { 1,  2,  3,  2,  1, },
                { 3,  2, -1,  2,  3, },
                { 1,  1,  0,  1,  1, },
                { 0,  0, -2,  1,  1, },
                { 3,  0,  0,  2,  4, },
            };

            FillSquare(d, datavalues);


            //kernel
            var kernelvalues = new float[,]
            {
                { 1,  2,  3, },
                { 3, -2,  1, },
                { 1,  1,  0, },
            };
            bias[0] = 0;

            FillKb(k, kernelvalues);

            //target:
            target.Fill(1f);

            //right target:
            var righttargervalues = new float[,]
            {
                { 21,  26,   8, },
                {  6,   9,  11, },
                {  5,  10,   1, },
            };

            var cc = new NaiveConvolutionCalculator(
                );

            cc.CalculateConvolutionWithIncrement(
                k,
                d,
                t
                );

            float maxDiff;
            var r = ArrayOperations.ValuesAreEqual(
                target,
                righttargervalues.ToLine(),
                float.Epsilon,
                out maxDiff
                );

            Assert.IsTrue(r);

        }

        [TestMethod]
        public void TestIncrementBias()
        {
            const int datadim = 5;
            const int datatotal = datadim * datadim;

            const int kerneldim = 3;
            const int kerneltotal = kerneldim * kerneldim;

            const int targetdim = datadim - kerneldim + 1;
            const int targettotal = targetdim * targetdim;

            var datasd = new Dimension(2, datadim, datadim);
            var kernelsd = new Dimension(2, kerneldim, kerneldim);
            var targetsd = new Dimension(2, targetdim, targetdim);

            var data = new float[datatotal];

            var kernel = new float[kerneltotal];
            var bias = new float[1];

            var target = new float[targettotal];

            var d = new ReferencedSquareFloat(
                datasd,
                data,
                0
                );

            var k = new ReferencedKernelBiasContainer(
                kernelsd,
                kernel,
                0,
                bias,
                0
                );


            var t = new ReferencedSquareFloat(
                targetsd,
                target,
                0
                );

            //data:
            var datavalues = new float[,]
            {
                { 1,  2,  3,  2,  1, },
                { 3,  2, -1,  2,  3, },
                { 1,  1,  0,  1,  1, },
                { 0,  0, -2,  1,  1, },
                { 3,  0,  0,  2,  4, },
            };

            FillSquare(d, datavalues);


            //kernel
            var kernelvalues = new float[,]
            {
                { 1,  2,  3, },
                { 3, -2,  1, },
                { 1,  1,  0, },
            };
            bias[0] = 1;

            FillKb(k, kernelvalues);

            //target:
            target.Fill(1f);

            //right target:
            var righttargervalues = new float[,]
            {
                { 22,  27,   9, },
                {  7,  10,  12, },
                {  6,  11,   2, },
            };

            var cc = new NaiveConvolutionCalculator(
                );

            cc.CalculateConvolutionWithIncrement(
                k,
                d,
                t
                );

            float maxDiff;
            var r = ArrayOperations.ValuesAreEqual(
                target,
                righttargervalues.ToLine(),
                float.Epsilon,
                out maxDiff
                );

            Assert.IsTrue(r);

        }

        [TestMethod]
        public void TestBackOverwrite()
        {
            const int datadim = 3;
            const int datatotal = datadim * datadim;

            const int kerneldim = 2;
            const int kerneltotal = kerneldim * kerneldim;

            const int targetdim = 4;
            const int targettotal = targetdim * targetdim;

            var datasd = new Dimension(2, datadim, datadim);
            var kernelsd = new Dimension(2, kerneldim, kerneldim);
            var targetsd = new Dimension(2, targetdim, targetdim);

            var data = new float[datatotal];

            var kernel = new float[kerneltotal];

            var target = new float[targettotal];

            var d = new ReferencedSquareFloat(
                datasd,
                data,
                0
                );

            var k = new ReferencedSquareFloat(
                kernelsd,
                kernel,
                0
                );


            var t = new ReferencedSquareFloat(
                targetsd,
                target,
                0
                );

            //data:
            var datavalues = new float[,]
            {
                { 1,  2,  3,  },
                { 3,  2, -1,  },
                { 1,  1,  0,  },
            };

            //-2   3
            // 2   1

            FillSquare(d, datavalues);


            //kernel
            var kernelvalues = new float[,]
            {
                { 1,  2, },
                { 3, -2, },
            };

            FillSquare(k, kernelvalues);

            //right target:
            var righttargervalues = new float[,]
            {
                {  1,   4,   7,   6 },
                {  6,  12,   8,  -8 },
                { 10,   3,  -5,   2 },
                {  3,   1,  -2,   0 },
            };

            var cc = new NaiveConvolutionCalculator(
                );

            cc.CalculateBackConvolutionWithOverwrite(
                k,
                d,
                t
                );

            float maxDiff;
            var r = ArrayOperations.ValuesAreEqual(
                target,
                righttargervalues.ToLine(),
                float.Epsilon,
                out maxDiff
                );

            Assert.IsTrue(r);

        }


        private void FillKb(
            IReferencedKernelBiasContainer target,
            float[,] datavalues
            )
        {
            if (target == null)
            {
                throw new ArgumentNullException("target");
            }
            if (datavalues == null)
            {
                throw new ArgumentNullException("datavalues");
            }
            if (target.Height != datavalues.GetLength(0))
            {
                throw new ArgumentException("target.Height != datavalues.GetLength(0)");
            }
            if (target.Width != datavalues.GetLength(1))
            {
                throw new ArgumentException("target.Width != datavalues.GetLength(1)");
            }

            for (var h = 0; h < datavalues.GetLength(0); h++)
            {
                for (var w = 0; w < datavalues.GetLength(1); w++)
                {
                    target.Kernel.SetValueFromCoordSafely(w, h, datavalues[h, w]);
                }
            }
        }

        private void FillSquare(
            IReferencedSquareFloat target,
            float[,] datavalues
            )
        {
            if (target == null)
            {
                throw new ArgumentNullException("target");
            }
            if (datavalues == null)
            {
                throw new ArgumentNullException("datavalues");
            }
            if (target.Height != datavalues.GetLength(0))
            {
                throw new ArgumentException("target.Height != datavalues.GetLength(0)");
            }
            if (target.Width != datavalues.GetLength(1))
            {
                throw new ArgumentException("target.Width != datavalues.GetLength(1)");
            }

            for (var h = 0; h < datavalues.GetLength(0); h++)
            {
                for (var w = 0; w < datavalues.GetLength(1); w++)
                {
                    target.SetValueFromCoordSafely(w, h, datavalues[h, w]);
                }
            }
        }
    }
}
