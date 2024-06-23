# Apparmor Example
## White List
![image](https://hackmd.io/_uploads/SJrKqgE8C.png)




```shell
#include <tunables/global>

/usr/local/bin/foo flags=(audit) {
    #include <abstractions/base>
    
}
```

![image](https://hackmd.io/_uploads/HyKCdgNI0.png)

Result:

![image](https://hackmd.io/_uploads/HytoTx48C.png)


```shell
#include <tunables/global>

/usr/local/bin/foo flags=(audit) {
    #include <abstractions/base>
    
    allow /tmp/read.txt r,
    allow /tmp/write.txt w,
}
```
Result:

![image](https://hackmd.io/_uploads/rkH5OPVLC.png)



## Black List
```shell
#include <tunables/global>

/usr/local/bin/foo flags=(audit) {
    #include <abstractions/base>

    file,
}
```
Result:

![image](https://hackmd.io/_uploads/HyCbhvVIC.png)

```shell
#include <tunables/global>

/usr/local/bin/foo flags=(audit) {
    #include <abstractions/base>

    file,
    
    deny /tmp/write.txt r,
}
```

Result:
![image](https://hackmd.io/_uploads/S1eBaDN8R.png)


```shell
#include <tunables/global>

/usr/local/bin/foo flags=(audit) {
    #include <abstractions/base>

    file,
    
    deny /tmp/write.txt r,
    
    allow /tmp/write.txt r,
}
```

Result:
![image](https://hackmd.io/_uploads/Bk7s6DV8A.png)



## Profile Example
Result:
![image](https://hackmd.io/_uploads/Hy9-k9V80.png)

```shell
#include <tunables/global>

# a comment naming the application to confine
/usr/local/bin/foo {
   #include <abstractions/base>

   capability setgid,
   network inet tcp,

   link /etc/sysconfig/foo -> /etc/foo.conf,
   /bin/mount            ux,
   /dev/{,u}random     r,
   /etc/ld.so.cache      r,
   /etc/foo/*            r,
   /lib/ld-*.so*         mr,
   /lib/lib*.so*         mr,
   /proc/[0-9]**         r,
   /usr/lib/**           mr,
   /tmp/                 r,
   /tmp/foo.pid          wr,
   /tmp/foo_test.txt     rw,
   /tmp/foo.*            lrw,
   /@{HOME}/.foo_file   rw,
   /@{HOME}/.foo_lock    kw,
   owner /shared/foo/** rw,
   /usr/local/bin/foobar       Cx -> foobar,
   /bin/**               Px -> bin_generic,

   # a comment about foo's local (children) profile for /usr/local/bin/foobar.

   profile foobar {
      /bin/bash          rmix,
      /bin/cat           rmix,
      /bin/more          rmix,
      /var/log/foobar*   rwl,
      /etc/foobar        r,
      /usr/local/bin/foobar r,
   }

  # foo's hat, bar.
   ^bar {
    /lib/ld-*.so*         mr,
    /usr/bin/bar          px,
    /var/spool/*          rwl,
   }
}
```

Result:
![image](https://hackmd.io/_uploads/r11AFq4U0.png)


## MediaStreamAddon Profile
```shell
#include <tunables/global>
/share/*/.qpkg/QDMS/bin/myupnpmediasvr flags=(audit) {
    #include <abstractions/base>
    signal,
    capability,
    network,

    allow /share/*/.system/log/myupnpmediasvr.log rw,
    allow /share/*/.system/dircfg.json rw,
    allow /share/*/.system/thumbnail/** r,
    allow /share/*/.system/music/** r,
    allow /share/*/.qpkg/QDMS/lib/* rm,
    allow /share/*/.qpkg/QDMS/etc/MyUPnPMediaSvr.conf rw,
    allow /share/*/.qpkg/QDMS/bin/UPnP_Profiles/* rwk,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/config/arm/* r,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/config/clientTypes.xml r,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/config/evansportHW/* r,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/config/x86/* r,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/config/x86VATranscode/* r,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/mymedia_cli ix,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/bin/mymediadbcmd ix,
    allow /share/*/.qpkg/MultimediaConsole/medialibrary/lib/* rm,
    allow /share/*/.qpkg/QDevelop/bin/libqcloud_wrap.so rm,
    allow /share/*/.qpkg/QDevelop/bin/qcloud_wrap.so rm,
    allow /share/*/.qpkg/*/CodexPackExt/gst-transcode-agent2 ix,
    allow /share/*/.qpkg/*/CodexPackExt/static/bin/ffmpeg ix,
    allow /share/*/.qpkg/*/CodexPackExt/static/bin/heic-thumb ix,
    allow /share/*/.qpkg/*/CodexPackExt/thumb_ffmpeg.sh ix,
    allow /usr/bin/ffmpeg ix,
    allow /usr/local/*/bin/ffmpeg ix,
    allow /usr/local/*/bin/ffprobe ix,
    allow /dev/pts/* rw,
    allow /dev/shm/sem.MYIDBS rw,
    allow /dev/shm/sem.MYIDBSSTOP rw,
    allow /dev/shm/sem.MYIDBSVC r,
    allow /dev/tty rw,
    allow /etc/nsswitch.conf rw,
    allow /etc/services r,
    allow /mnt/ext/opt/gconv/gconv-modules r,
    allow /mnt/ext/opt/mariadb/lib/libmysqlclient.so.* rm,
    allow /mnt/ext/opt/mariadb/share/charsets/Index.xml r,
    allow /mnt/HDA_ROOT/.config/group r,
    allow /mnt/HDA_ROOT/.config/medialibrary.conf rw,
    allow /mnt/HDA_ROOT/.config/MyUPnPMediaSvr.conf rw,
    allow /mnt/HDA_ROOT/.config/passwd r,
    allow /mnt/HDA_ROOT/.config/qpkg.conf r,
    allow /mnt/HDA_ROOT/.config/shadow r,
    allow /mnt/HDA_ROOT/.config/smb.conf r,
    allow /mnt/HDA_ROOT/.config/uLinux.conf r,
    allow /proc/@{pid}/fd/{,**} r,
    allow /proc/@{pid}/net/arp* r,
    allow /sys/devices/pci*/*/*/net/*/address r,
    allow /tmp/.mldircfg.lock rwk,
    allow /tmp/.myupnpmediasvr.nvs w,
    allow /tmp/medialibrary/mldebug.conf r,
    allow /tmp/mymediadbserver-sysvar.lock rwk,
    allow /tmp/myupnpmediasvr.lock rwk,
    allow /var/lock/*.lck rwk,
    allow /var/log/ini_config.log ra,
    allow /var/log/myupnpmediasvr_crit.log rw,

    allow /bin/busybox ix,
    allow /sbin/getcfg ix,
    allow /sbin/setcfg ix,
    allow /bin/sh ix,
    allow /bin/ip ix,

    allow /share/** r,
}

```
