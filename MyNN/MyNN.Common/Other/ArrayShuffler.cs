using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Randomizer;

namespace MyNN.Common.Other
{
    public class ArrayShuffler<T> : IEnumerable<T>
    {
        private readonly IRandomizer _randomizer;
        private readonly IEnumerable<T> _e;

        public ArrayShuffler(
            IRandomizer randomizer,
            IEnumerable<T> e 
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (e == null)
            {
                throw new ArgumentNullException("e");
            }

            _randomizer = randomizer;
            _e = e;
        }

        public IEnumerator<T> GetEnumerator()
        {
            var list = new List<T>();

            foreach (var i in _e)
            {
                list.Add(i);
            }

            for (var i = 0; i < list.Count - 1; i++)
            {
                if (_randomizer.Next() >= 0.5f)
                {
                    var newIndex = _randomizer.Next(list.Count);

                    var tmp = list[i];
                    list[i] = list[newIndex];
                    list[newIndex] = tmp;
                }
            }

            foreach (var l in list)
            {
                yield
                    return l;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
