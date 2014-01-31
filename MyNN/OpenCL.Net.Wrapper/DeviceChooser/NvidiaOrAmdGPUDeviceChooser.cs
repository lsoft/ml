using System;
using System.Linq;


namespace OpenCL.Net.Wrapper.DeviceChooser
{
    public class NvidiaOrAmdGPUDeviceChooser : IDeviceChooser
    {
        public void ChooseDevice(
            out DeviceType choosedDeviceType,
            out Device choosedDevice)
        {
            ErrorCode error;

            var platforms = Cl.GetPlatformIDs(out error);
            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Unable to retrieve an OpenCL Device, error was: {0}!",
                        error));
            }

            foreach (var platform in platforms)
            {
                var deviceIds = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out error);
                if (deviceIds.Any())
                {
                    foreach (var device in deviceIds)
                    {
                        var vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out error).ToString();
                        var uvendor = vendor.ToUpper();

                        //Console.WriteLine(uvendor);

                        if (uvendor.Contains("NVIDIA") || uvendor.Contains("ADVANCED MICRO DEVICES"))
                        {
                            Console.WriteLine(
                                "Choosed vendor: {0}",
                                uvendor);

                            choosedDevice = deviceIds.First();
                            choosedDeviceType = DeviceType.Gpu;
                            return;
                        }
                    }
                }
            }

            throw new InvalidOperationException("There is no NVIDIA/ATI/AMD GPU device");
        }
    }
}