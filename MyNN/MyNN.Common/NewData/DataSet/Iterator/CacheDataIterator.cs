using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Threading;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public class CacheDataIterator : IDataIterator
    {
        private readonly int _cacheSize;
        private readonly IDataIterator _iterator;
        
        private ConcurrentQueue<IDataItem> _queue;
        private Thread _t;
        private bool _isNeedToStop;

        public IDataItem Current
        {
            get;
            private set;
        }

        object IEnumerator.Current
        {
            get
            {
                return Current;
            }
        }

        public int Count
        {
            get
            {
                return
                    _iterator.Count;
            }
        }

        public CacheDataIterator(
            int cacheSize,
            IDataIterator iterator
            )
        {
            if (iterator == null)
            {
                throw new ArgumentNullException("iterator");
            }

            _cacheSize = cacheSize;
            _iterator = iterator;

            _queue = new ConcurrentQueue<IDataItem>();

            StartBackground();
        }

        public bool MoveNext()
        {
            IDataItem newItem;
            while (!_queue.TryDequeue(out newItem))
            {
                Thread.Sleep(20);
            }

            var result = newItem != null; //если null - коллекция закончилась
            this.Current = newItem;

            return result;
        }

        public void Reset()
        {
            StopBackground();

            _iterator.Reset();

            StartBackground();
        }

        public void Dispose()
        {
            StopBackground();

            _iterator.Dispose();
        }

        private void StartBackground()
        {
            _queue = new ConcurrentQueue<IDataItem>();

            _isNeedToStop = false;
            _t = new Thread(InternalStayCacheInActualState);
            _t.IsBackground = true;
            _t.Start();
        }

        private void StopBackground()
        {
            _isNeedToStop = true;
            _t.Join();
        }

        private void InternalStayCacheInActualState(
            )
        {
            var maycontinue = true;
            while (!_isNeedToStop)
            {
                while (maycontinue && _queue.Count < _cacheSize)
                {
                    //добавляем в кеш значения

                    if (_iterator.MoveNext())
                    {
                        var cv = _iterator.Current;
                        this._queue.Enqueue(cv);
                    }
                    else
                    {
                        //кончились значения
                        this._queue.Enqueue(null);
                        maycontinue = false;
                        break;
                    }
                }

                Thread.Sleep(25);
            }
        }
    }
}