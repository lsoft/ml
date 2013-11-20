﻿#region License and Copyright Notice

//OpenCL.Net: .NET bindings for OpenCL

//Copyright (c) 2010 Ananth B.
//All rights reserved.

//The contents of this file are made available under the terms of the
//Eclipse Public License v1.0 (the "License") which accompanies this
//distribution, and is available at the following URL:
//http://www.opensource.org/licenses/eclipse-1.0.php

//Software distributed under the License is distributed on an "AS IS" basis,
//WITHOUT WARRANTY OF ANY KIND, either expressed or implied. See the License for
//the specific language governing rights and limitations under the License.

//By using this software in any fashion, you are agreeing to be bound by the
//terms of the License.

#endregion

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCL.Net
{
    public static partial class Cl
    {
        internal static InfoBuffer GetInfo<THandleType, TEnumType>(
            GetInfoDelegate<THandleType, TEnumType> method, THandleType handle, TEnumType name, out ErrorCode error)
        {
            IntPtr paramSize;
            error = method(handle, name, IntPtr.Zero, InfoBuffer.Empty, out paramSize);
            if (error != ErrorCode.Success)
            {
                return InfoBuffer.Empty;
            }

            var buffer = new InfoBuffer(paramSize);
            error = method(handle, name, paramSize, buffer, out paramSize);
            if (error != ErrorCode.Success)
            {
                return InfoBuffer.Empty;
            }

            return buffer;
        }

        internal static InfoBuffer GetInfo<THandle1Type, THandle2Type, TEnumType>(
            GetInfoDelegate<THandle1Type, THandle2Type, TEnumType> method, THandle1Type handle1, THandle2Type handle2, TEnumType name, out ErrorCode error)
        {
            IntPtr paramSize;
            error = method(handle1, handle2, name, IntPtr.Zero, InfoBuffer.Empty, out paramSize);
            if (error != ErrorCode.Success)
            {
                return InfoBuffer.Empty;
            }

            var buffer = new InfoBuffer(paramSize);
            error = method(handle1, handle2, name, paramSize, buffer, out paramSize);
            if (error != ErrorCode.Success)
            {
                return InfoBuffer.Empty;
            }

            return buffer;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct InfoBuffer : IDisposable
        {
            private static readonly InfoBuffer _empty = new InfoBuffer
                                                        {
                                                            _buffer = IntPtr.Zero
                                                        };

            private IntPtr _buffer;

            public InfoBuffer(IntPtr size)
            {
                _buffer = Marshal.AllocHGlobal(size);
            }

            internal IntPtr Address
            {
                get
                {
                    return _buffer;
                }
            }

            public T CastTo<T>() where T : struct
            {
                return _buffer.ElementAt<T>(0);
            }

            public T CastTo<T>(int index) where T : struct
            {
                return _buffer.ElementAt<T>(index);
            }

            public IEnumerable<T> CastToEnumerable<T>(IEnumerable<int> indices) where T : struct
            {
                foreach (int index in indices)
                {
                    yield return _buffer.ElementAt<T>(index);
                }
            }

            public override string ToString()
            {
                return Marshal.PtrToStringAnsi(_buffer);
            }

            public static InfoBuffer Empty
            {
                get
                {
                    return _empty;
                }
            }

            #region IDisposable Members

            public void Dispose()
            {
                if (_buffer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_buffer);
                    _buffer = IntPtr.Zero;
                }
            }

            #endregion
        }

        public struct PinnedObject : IDisposable
        {
            private readonly GCHandle _handle;

            internal PinnedObject(object obj)
            {
                _handle = GCHandle.Alloc(obj, GCHandleType.Pinned);
            }

            #region IDisposable Members

            public void Dispose()
            {
                _handle.Free();
            }

            #endregion

            public static implicit operator IntPtr(PinnedObject pinned)
            {
                return pinned._handle.AddrOfPinnedObject();
            }
        }

        public static IntPtr Increment(this IntPtr ptr, int cbSize)
        {
            return new IntPtr(ptr.ToInt64() + cbSize);
        }

        public static IntPtr Increment<T>(this IntPtr ptr)
        {
            return ptr.Increment(Marshal.SizeOf(typeof (T)));
        }

        public static T ElementAt<T>(this IntPtr ptr, int index) where T : struct
        {
            Type resultType = typeof (T);
            resultType = resultType.IsEnum ? Enum.GetUnderlyingType(resultType) : resultType;

            var offset = Marshal.SizeOf(resultType)*index;
            var offsetPtr = ptr.Increment(offset);

            return (T) Marshal.PtrToStructure(offsetPtr, resultType);
        }

        public static ErrorCode OnError(this ErrorCode error, ErrorCode errorCode, Action<ErrorCode> action)
        {
            if (error == errorCode)
            {
                action(error);
            }

            return error;
        }

        public static ErrorCode OnAnyError(this ErrorCode error, Action<ErrorCode> action)
        {
            if (error != ErrorCode.Success)
            {
                action(error);
            }

            return error;
        }

        public static PinnedObject Pin(this object obj)
        {
            return new PinnedObject(obj);
        }

        public static T[] InitializeArray<T>(this T[] arr) where T : new()
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = new T();
            }

            return arr;
        }
    }
}