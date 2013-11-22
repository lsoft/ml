using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace MyNN
{
    //класс не проверен в работе
    //public class ThreadSafetyHelper
    //{
    //    // AddToTotal safely adds a value to the running total.
    //    public float SafelyIncrement(
    //        ref float increment)
    //    {
    //        float valueToUpdate = 0;

    //        float initialValue, computedValue;
    //        do
    //        {
    //            // Save the current running total in a local variable.
    //            initialValue = valueToUpdate;

    //            // Add the new value to the running total.
    //            computedValue = initialValue + increment;

    //            // CompareExchange compares totalValue to initialValue. If
    //            // they are not equal, then another thread has updated the
    //            // running total since this loop started. CompareExchange
    //            // does not update totalValue. CompareExchange returns the
    //            // contents of totalValue, which do not equal initialValue,
    //            // so the loop executes again.
    //        } while (Math.Abs(initialValue - Interlocked.CompareExchange(
    //            ref valueToUpdate, computedValue, initialValue)) > float.Epsilon);

    //        // If no other thread updated the running total, then 
    //        // totalValue and initialValue are equal when CompareExchange
    //        // compares them, and computedValue is stored in totalValue.
    //        // CompareExchange returns the value that was in totalValue
    //        // before the update, which is equal to initialValue, so the 
    //        // loop ends.

    //        // The function returns computedValue, not totalValue, because
    //        // totalValue could be changed by another thread between
    //        // the time the loop ends and the function returns.
    //        return computedValue;
    //    }

    //}
}
