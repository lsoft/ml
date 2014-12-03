using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MyNNConsoleApp.CTRP
{
    internal class Item
    {
        public List<object> Objects
        {
            get;
            set;
        }

        public bool? Clicked
        {
            get;
            set;
        }

        public DateTime Datetime
        {
            get;
            set;
        }

        public int C1
        {
            get;
            set;
        }

        public int BannerPos
        {
            get;
            set;
        }

        public int SiteId
        {
            get;
            set;
        }

        public int SiteDomain
        {
            get;
            set;
        }

        public int SiteCategory
        {
            get;
            set;
        }

        public int AppId
        {
            get;
            set;
        }

        public int AppDomain
        {
            get;
            set;
        }

        public int AppCategory
        {
            get;
            set;
        }

        public int DeviceId
        {
            get;
            set;
        }

        public int DeviceIp
        {
            get;
            set;
        }

        public int DeviceModel
        {
            get;
            set;
        }

        public int DeviceType
        {
            get;
            set;
        }

        public int DeviceConnType
        {
            get;
            set;
        }

        public int C14
        {
            get;
            set;
        }

        public int C15
        {
            get;
            set;
        }

        public int C16
        {
            get;
            set;
        }

        public int C17
        {
            get;
            set;
        }

        public int C18
        {
            get;
            set;
        }

        public int C19
        {
            get;
            set;
        }

        public int C20
        {
            get;
            set;
        }

        public int C21
        {
            get;
            set;
        }

        public Item(
            bool? clicked,
            DateTime datetime,
            int c1,
            int bannerPos,
            int siteId,
            int siteDomain,
            int siteCategory,
            int appId,
            int appDomain,
            int appCategory,
            int deviceId,
            int deviceIp,
            int deviceModel,
            int deviceType,
            int deviceConnType,
            int c14,
            int c15,
            int c16,
            int c17,
            int c18,
            int c19,
            int c20,
            int c21
            )
        {
            Objects = new List<object>();

            Clicked = clicked;
            Objects.Add(clicked);

            Datetime = datetime;
            Objects.Add(datetime);

            C1 = c1;
            Objects.Add(c1);

            BannerPos = bannerPos;
            Objects.Add(bannerPos);

            SiteId = siteId;
            Objects.Add(siteId);

            SiteDomain = siteDomain;
            Objects.Add(siteDomain);

            SiteCategory = siteCategory;
            Objects.Add(siteCategory);

            AppId = appId;
            Objects.Add(appId);

            AppDomain = appDomain;
            Objects.Add(appDomain);

            AppCategory = appCategory;
            Objects.Add(appCategory);

            DeviceId = deviceId;
            Objects.Add(deviceId);

            DeviceIp = deviceIp;
            Objects.Add(deviceIp);

            DeviceModel = deviceModel;
            Objects.Add(deviceModel);

            DeviceType = deviceType;
            Objects.Add(deviceType);

            DeviceConnType = deviceConnType;
            Objects.Add(deviceConnType);

            C14 = c14;
            Objects.Add(c14);

            C15 = c15;
            Objects.Add(c15);

            C16 = c16;
            Objects.Add(c16);

            C17 = c17;
            Objects.Add(c17);

            C18 = c18;
            Objects.Add(c18);

            C19 = c19;
            Objects.Add(c19);

            C20 = c20;
            Objects.Add(c20);

            C21 = c21;
            Objects.Add(c21);

        }

        public void WriteTo(BinaryWriter bw)
        {
            bw.Write(Clicked == null ? 0 : 1);
            bw.Write(Clicked == null ? false : Clicked.Value);

            bw.Write(Datetime.Ticks);
            bw.Write(C1);
            bw.Write(BannerPos);
            bw.Write(SiteId);
            bw.Write(SiteDomain);
            bw.Write(SiteCategory);
            bw.Write(AppId);
            bw.Write(AppDomain);
            bw.Write(AppCategory);
            bw.Write(DeviceId);
            bw.Write(DeviceIp);
            bw.Write(DeviceModel);
            bw.Write(DeviceType);
            bw.Write(DeviceConnType);
            bw.Write(C14);
            bw.Write(C15);
            bw.Write(C16);
            bw.Write(C17);
            bw.Write(C18);
            bw.Write(C19);
            bw.Write(C20);
            bw.Write(C21);
        }

        public void WriteSizeTo(BinaryWriter bw)
        {
            if (bw == null)
            {
                throw new ArgumentNullException("bw");
            }

            using (var ms = new MemoryStream())
            {
                using (var bw2 = new BinaryWriter(ms, Encoding.Default, true))
                {
                    this.WriteTo(bw2);
                }

                var size = ms.Length;
                bw.Write(size);

                Console.WriteLine(
                    "item size = {0}",
                    size);
            }
        }

        public static Item Read(BinaryReader br)
        {
            var cl0 = br.ReadInt32();
            var cl1 = br.ReadBoolean();

            var ticks = br.ReadInt64();
            var c1 = br.ReadInt32();
            var banner_pos = br.ReadInt32();
            var site_id = br.ReadInt32();
            var site_domain = br.ReadInt32();
            var site_category = br.ReadInt32();
            var app_id = br.ReadInt32();
            var app_domain = br.ReadInt32();
            var app_category = br.ReadInt32();
            var device_id = br.ReadInt32();
            var device_ip = br.ReadInt32();
            var device_model = br.ReadInt32();
            var device_type = br.ReadInt32();
            var device_conn_type = br.ReadInt32();
            var c14 = br.ReadInt32();
            var c15 = br.ReadInt32();
            var c16 = br.ReadInt32();
            var c17 = br.ReadInt32();
            var c18 = br.ReadInt32();
            var c19 = br.ReadInt32();
            var c20 = br.ReadInt32();
            var c21 = br.ReadInt32();

            var item = new Item(
                cl0 == 0 ? (bool?)null : cl1,
                new DateTime(ticks),
                c1,
                banner_pos,
                site_id,
                site_domain,
                site_category,
                app_id,
                app_domain,
                app_category,
                device_id,
                device_ip,
                device_model,
                device_type,
                device_conn_type,
                c14,
                c15,
                c16,
                c17,
                c18,
                c19,
                c20,
                c21
                );

            return item;
        }

    }

}