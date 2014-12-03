using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet.ItemLoader;

namespace MyNNConsoleApp.CTRP
{
    public class CRPDataItemLoader : IDataItemLoader
    {
        private const string TxtBin = "__combined.txtbin";

        private readonly string _folderPath;
        private readonly int _desiredCount;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly FileStream _filestream;

        private readonly float[] _hourfloats = new float[24];
        private readonly float[] _dowfloats = new float[7];
        private readonly Dictionary<int, float[]> _maxfloats = new Dictionary<int, float[]>();

        private long _itemSize;

        public int Count
        {
            get;
            private set;
        }

        public CRPDataItemLoader(
            string folderPath,
            string dataFileName,
            int desiredCount,
            IDataItemFactory dataItemFactory
            )
        {
            if (folderPath == null)
            {
                throw new ArgumentNullException("folderPath");
            }
            if (dataFileName == null)
            {
                throw new ArgumentNullException("dataFileName");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            var datafilepath = Path.Combine(folderPath, dataFileName);
            if (!File.Exists(datafilepath))
            {
                throw new ArgumentException("!File.Exists: " + datafilepath);
            }
            var txtbinfilepath = Path.Combine(folderPath, TxtBin);
            if (!File.Exists(txtbinfilepath))
            {
                throw new ArgumentException("!File.Exists: " + txtbinfilepath);
            }

            _folderPath = folderPath;
            _desiredCount = desiredCount;
            _dataItemFactory = dataItemFactory;
            _filestream = File.OpenRead(datafilepath);

            ReadFirst();
        }

        public void Normalize(float bias = 0)
        {
            throw new NotSupportedException("Пока не поддерживается");
        }

        public void GNormalize()
        {
            throw new NotSupportedException("Пока не поддерживается");
        }

        public IDataItem Load(int index)
        {
            if (index < 0)
            {
                throw new ArgumentException("index < 0");
            }
            if (index >= this.Count)
            {
                throw new ArgumentException("index >= this.Count");
            }

            _filestream.Position = 12 + index*_itemSize;

            using (var br = new BinaryReader(_filestream, Encoding.Default, true))
            {
                var i = Item.Read(br);

                var result = ConvertToDataItem(i);

                return result;
            }
        }

        #region private code

        private IDataItem ConvertToDataItem(
            Item i
            )
        {
            var input = new FloatContainer();
            var output = new List<float>();

            var clicked = i.Clicked == true;
            if (clicked)
            {
                output.Add(1f);
                output.Add(0f);
            }
            else
            {
                output.Add(0f);
                output.Add(1f);
            }


            var hour = i.Datetime.Hour;
            _hourfloats[hour] = 1f;
            input.AddRange(_hourfloats);
            _hourfloats[hour] = 0f;

            var dayofweek = (int)i.Datetime.DayOfWeek;
            _dowfloats[dayofweek] = 1f;
            input.AddRange(_dowfloats);
            _dowfloats[dayofweek] = 0f;

            foreach (var dd in new int[] { 2, 3, 6, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22 })
            {
                var v = (int)(i.Objects[dd]);

                _maxfloats[dd][v] = 1f;
                input.AddRange(_maxfloats[dd]);
                _maxfloats[dd][v] = 0f;
            }

            var result = _dataItemFactory.CreateDataItem(input.GetArray().ToArray(), output.ToArray());

            return
                result;
        }

        private void ReadFirst()
        {
            using (var br = new BinaryReader(_filestream, Encoding.Default, true))
            {
                this.Count = Math.Min(_desiredCount, br.ReadInt32());
                this._itemSize = br.ReadInt64();
            }

            var txtbinfilepath = Path.Combine(_folderPath, TxtBin);
            var liter = File.ReadLines(txtbinfilepath);

            string headerstring = null;
            string maxestring = null;

            var iii = 0;
            foreach (var l in liter)
            {
                if (iii == 0)
                {
                    headerstring = l;
                }
                if (iii == 1)
                {
                    maxestring = l;
                }
                if (iii == 2)
                {
                    break;
                }

                iii++;
            }

            var maxes = maxestring.Split(',').Skip(1).ToList().ConvertAll(j => int.Parse(j));

            foreach (var dd in new int[] { 2, 3, 6, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22 })
            {
                var max = maxes[dd];
                var floats = new float[max];
                _maxfloats.Add(dd, floats);
            }
        }

        private class FloatContainer
        {
            private static int _arrayLength = 0;

            private float[] _array = null;
            private int _currentSize = 0;

            public FloatContainer()
            {
                _array = new float[_arrayLength];
                _currentSize = 0;
            }

            public void AddRange(float[] part)
            {
                if (part == null)
                {
                    throw new ArgumentNullException("part");
                }

                var newSize = _currentSize + part.Length;
                if (newSize > _array.Length)
                {
                    var newAllocatedSize = (int)(newSize * 1.4);

                    var newArray = new float[newAllocatedSize];
                    Array.Copy(_array, 0, newArray, 0, _currentSize);
                    Array.Copy(part, 0, newArray, _currentSize, part.Length);

                    _array = newArray;
                    _currentSize = newSize;
                }
                else
                {
                    Array.Copy(part, 0, _array, _currentSize, part.Length);
                    _currentSize += part.Length;
                }
            }

            public float[] GetArray()
            {
                if (_arrayLength == 0)
                {
                    _arrayLength = _currentSize;
                }

                float[] result;

                if (_currentSize == _array.Length)
                {
                    result = _array;
                }
                else
                {
                    result = new float[_currentSize];
                    Array.Copy(_array, 0, result, 0, _currentSize);
                }

                return
                    result;
            }

        }

        #endregion

        #region obsolete

        //public static IDataSet GetDataSet(
        //    string folderPath,
        //    int count,
        //    IDataItemFactory dataItemFactory
        //    )
        //{
        //    if (folderPath == null)
        //    {
        //        throw new ArgumentNullException("folderPath");
        //    }
        //    if (dataItemFactory == null)
        //    {
        //        throw new ArgumentNullException("dataItemFactory");
        //    }

        //    var liter = File.ReadLines("DATA #1/__combined.txtbin");

        //    string headerstring = null;
        //    string maxestring = null;

        //    var iii = 0;
        //    foreach (var l in liter)
        //    {
        //        if (iii == 0)
        //        {
        //            headerstring = l;
        //        }
        //        if (iii == 1)
        //        {
        //            maxestring = l;
        //        }
        //        if (iii == 2)
        //        {
        //            break;
        //        }

        //        iii++;
        //    }

        //    var maxes = maxestring.Split(',').Skip(1).ToList().ConvertAll(j => int.Parse(j));

        //    var dataitems = new List<IDataItem>();

        //    var hourfloats = new float[24];
        //    var dowfloats = new float[7];

        //    var maxfloats = new Dictionary<int, float[]>();
        //    foreach (var dd in new int[] { 2, 3, 6, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22 })
        //    {
        //        var max = maxes[dd];
        //        var floats = new float[max];
        //        maxfloats.Add(dd, floats);
        //    }


        //    using (var br = new BinaryReader(new FileStream(folderPath, FileMode.Open, FileAccess.Read)))
        //    {
        //        for (var cc = 0; cc < count; cc++)
        //        {
        //            if (br.BaseStream.Position == br.BaseStream.Length)
        //            {
        //                break;
        //            }

        //            var i = Item.Read(br);


        //            var input = new FloatContainer();
        //            var output = new List<float>();

        //            var clicked = i.Clicked == true;
        //            if (clicked)
        //            {
        //                output.Add(1f);
        //                output.Add(0f);
        //            }
        //            else
        //            {
        //                output.Add(0f);
        //                output.Add(1f);
        //            }


        //            var hour = i.Datetime.Hour;
        //            hourfloats[hour] = 1f;
        //            input.AddRange(hourfloats);
        //            hourfloats[hour] = 0f;

        //            var dayofweek = (int) i.Datetime.DayOfWeek;
        //            dowfloats[dayofweek] = 1f;
        //            input.AddRange(dowfloats);
        //            dowfloats[dayofweek] = 0f;

        //            foreach (var dd in new int[] {2, 3, 6, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22})
        //            {
        //                var v = (int) (i.Objects[dd]);

        //                maxfloats[dd][v] = 1f;
        //                input.AddRange(maxfloats[dd]);
        //                maxfloats[dd][v] = 0f;
        //            }

        //            var dataitem = dataItemFactory.CreateDataItem(input.GetArray().ToArray(), output.ToArray());
        //            dataitems.Add(dataitem);

        //            if (cc % 10000 == 0)
        //            {
        //                if (count == int.MaxValue)
        //                {
        //                    Console.Write(
        //                        "{0} %     ",
        //                        (long)br.BaseStream.Position * 100L / br.BaseStream.Length);
        //                }
        //                else
        //                {
        //                    Console.Write(
        //                        "{0} %  ({1} / {2})   ",
        //                        (long)cc * 100L / count,
        //                        cc,
        //                        count);
        //                }
        //                Console.SetCursorPosition(0, Console.CursorTop);

        //                GC.Collect(2);
        //            }
        //        }
        //    }

        //    //File.AppendAllLines(
        //    //    "___new.txt",
        //    //    dataitems.ConvertAll(j => string.Join(" ", j.Input))
        //    //    );

        //    return
        //        new DataSet(dataitems);
        //}

        #endregion
    }
}