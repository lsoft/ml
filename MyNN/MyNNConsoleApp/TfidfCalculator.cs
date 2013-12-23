using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNNConsoleApp
{
    /// <summary>
    /// калькулятор tfidf по корпусу слов (применялся в kaggle partly sunning,
    /// поэтому для других задач ВОЗМОЖНО придется его переосмыслить и дорабоотать)
    /// </summary>
    public class TfidfCalculator
    {
        private readonly Dictionary<string, Dictionary<string, int>> _corpus;
        private readonly Dictionary<string, int> _corpus2;


        public TfidfCalculator(List<string> corpus)
        {
            if (corpus == null)
            {
                throw new ArgumentNullException("corpus");
            }

            _corpus = 
                corpus
                    .Distinct()
                    .ToDictionary(
                        k => k,
                        j => j.Split(new[] { " " }, StringSplitOptions.RemoveEmptyEntries)
                            .Distinct()
                            .ToDictionary(z => z, w => 1));

            List<Dictionary<string, int>> prepared = corpus
                .ConvertAll(j => j.Split(new[] {" "}, StringSplitOptions.RemoveEmptyEntries).Distinct().ToDictionary(k => k, l => 1))
                .ToList();

            //prepared.ForEach(j => j.Keys.ToList().ForEach(k => string.Intern(k)));

            List<string> words = new List<string>();
            prepared.ForEach(j => words.AddRange(j.Keys.Distinct()));
            words = words.Distinct().ToList();

            _corpus2 = words.ToDictionary(
                a => a,
                j => prepared.AsParallel().Count(k => k.ContainsKey(j)));
        }

        public Dictionary<string, float> GetTFIDF(string document)
        {
            var result = new Dictionary<string, float>();

            var dwl = document.Split(new [] { " " }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var w in dwl)
            {
                var tfidf = float.MaxValue;

                //if (w.ToCharArray().All(j => !Char.IsDigit(j)))
                {
                    //слово не содержит цифирь

                    var tf = dwl.Count(j => j == w) / (double)dwl.Length;
                    var idf = Math.Log(
                        _corpus.Count/
                            (_corpus2.ContainsKey(w) ? _corpus2[w] : 0.0)
                            //(double)_corpus
                            //.AsParallel()
                            //.Where(j => j.Key != document)
                            //.Count(j => j.Value.ContainsKey(w))
                        );

                    if (!double.IsInfinity(tf) && !double.IsNaN(tf)
                        && !double.IsInfinity(idf) && !double.IsNaN(idf))
                    {
                        tfidf = (float) (tf*idf);
                    }
                }

                if (result.ContainsKey(w))
                {
                    if (result[w] < tfidf)
                    {
                        result[w] = tfidf;
                    }
                }
                else
                {
                    result.Add(w, tfidf);
                }
            }

            return result;
        }
    }
}
