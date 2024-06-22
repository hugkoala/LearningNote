# Apparmor Example
## Black List
```shell
```
## White List
```shell
```
## Profile Example
```shell
```
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
