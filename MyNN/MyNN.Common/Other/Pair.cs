using System;

namespace MyNN.Common.Other
{
    /// <summary>
    /// Класс, объединяющий в себе пару объектов
    /// Иногда бывает полезно объединить два значения в пару, но частный класс городить не хочется
    /// Можно использовать этот класс
    /// </summary>
    /// <typeparam name="T">Тип первого параметра</typeparam>
    /// <typeparam name="U">Тип второго параметра</typeparam>
    [Serializable]
    public class Pair<T, U>
    {
        /// <summary>
        /// Первый параметр
        /// </summary>
        public T First
        {
            get;
            set;
        }

        /// <summary>
        /// Второй параметр
        /// </summary>
        public U Second
        {
            get;
            set;
        }

        public Pair()
        {
        }

        public Pair(Pair<T, U> p)
        {
            First = p.First;
            Second = p.Second;
        }

        public Pair(T f, U s)
        {
            First = f;
            Second = s;
        }

    }
}
