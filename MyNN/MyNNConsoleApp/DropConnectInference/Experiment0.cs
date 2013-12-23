using System;

namespace MyNNConsoleApp.DropConnectInference
{
    public class Experiment0
    {
        public static void Execute()
        {
            var n = 6;
            var p = 0.5f;

            var wv =
                new[]
                {
                    0.2f,
                    0.3f,
                    0.4f,
                    0.5f,
                    0.6f,
                    0.2f,
                };

            const int sumcount = 100000;

            {
                var wvbino = new WVBino(p, n, wv);

                var u_bino_sum = 0.0;
                for (var sumindex = 0; sumindex < sumcount; sumindex++)
                {
                    u_bino_sum += wvbino.GetU();
                }

                Console.WriteLine("avg u through wvbino = {0}", u_bino_sum/(double) sumcount);
            }

            {
                var wvnormal_mean = 0.0;
                for (var cc = 0; cc < wv.Length; cc++)
                {
                    wvnormal_mean += wv[cc];
                }
                wvnormal_mean *= p;

                var wvnormal_var = 0.0;
                for (var cc = 0; cc < wv.Length; cc++)
                {
                    wvnormal_var += wv[cc]*wv[cc];
                }
                wvnormal_var *= p*(1.0f - p);

                var wvnorma = new MyNormal(wvnormal_mean, Math.Sqrt(wvnormal_var));

                var u_norma_sum = 0.0;
                for (var sumindex = 0; sumindex < sumcount; sumindex++)
                {
                    u_norma_sum += wvnorma.Sample();
                }

                Console.WriteLine("avg u through wvnorma= {0}", u_norma_sum/(double) sumcount);
            }

            //var mybino = new MyBino(p, n);
            //var bino = new Binomial(p, n);
            //var norma = new Normal(n * p, Math.Sqrt(n * p * (1f - p)));

            //var mb3 = mybino.Probability(3);
            //var b3 = bino.Probability(3);
            //var n3 = norma.Density(3);

            //var mb4 = mybino.Probability(4);
            //var b4 = bino.Probability(4);
            //var n4 = norma.Density(4);

            //var mb5 = mybino.Probability(5);
            //var b5 = bino.Probability(5);
            //var n5 = norma.Density(5);

            //Console.WriteLine("{0} {1} {2}", mb3, b3, n3);
            //Console.WriteLine("{0} {1} {2}", mb4, b4, n4);
            //Console.WriteLine("{0} {1} {2}", mb5, b5, n5);

            return;
        }
    }
}