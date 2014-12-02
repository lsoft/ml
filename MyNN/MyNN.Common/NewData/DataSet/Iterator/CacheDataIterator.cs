using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Threading;
using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public class CacheDataIterator : IDataIterator
    {
        private readonly int _cacheSize;
        private readonly IDataIterator _iterator;
        
        private readonly ConcurrentStack<IDataItem> _stack;
        
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

            _stack = new ConcurrentStack<IDataItem>();

            StartBackground();
        }

        public bool MoveNext()
        {
            IDataItem newItem;
            while (!_stack.TryPop(out newItem))
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
            StartBackground();
        }

        public void Dispose()
        {
            StopBackground();
        }

        private void StartBackground()
        {
            _stack.Clear();

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
            while (_isNeedToStop)
            {
                while (_stack.Count < _cacheSize)
                {
                    //добавляем в кеш значения

                    if (_iterator.MoveNext())
                    {
                        this._stack.Push(_iterator.Current);
                    }
                    else
                    {
                        //кончились значения
                        this._stack.Push(null);
                    }
                }

                Thread.Sleep(25);
            }
        }
    }
}