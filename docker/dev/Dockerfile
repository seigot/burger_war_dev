ARG CORE_VERSION=latest
FROM burger-war-core:$CORE_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# ドライバのインストールで止まる場合がある為、予めインストール
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        keyboard-configuration \
    && rm -rf /var/lib/apt/lists/*

# 調査用のコマンドをインストール
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        lshw \
        kmod \
        lsof \
    && rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------------------------
# 開発PC用のドライバをインストールする
# （Gazeboが起動しない場合やGPUを使いたい場合は、ここで必要なドライバをインストールして下さい）
#-------------------------------------------------------------------------------------------------
# ubuntu-driversによる自動インストールの例
#-------------------------------------------------
# RUN apt-get update -q && apt-get install -y --no-install-recommends \
#     ubuntu-drivers-common \
#     && rm -rf /var/lib/apt/lists/* \
#     && ubuntu-drivers autoinstall

#===================================================================================
# PCやグラフィックボードによっては、ubuntu-driversが失敗する場合もあります。
# その場合、以下の例を参考に、開発PCにインストールされているバージョンのドライバを
# 個別にインストールして下さい。
#===================================================================================

# NVIDIAのドライバをインストールする例 １) Ubuntuのデフォルトバージョンをインストール
#-------------------------------------------------
# RUN sudo apt-get update -q && sudo apt-get install -y \
#     nvidia-driver-450 \
#     && rm -rf /var/lib/apt/lists/*

# NVIDIAのドライバをインストールする例 ２) バージョンを指定してインストール
# 参考：https://gitlab.com/nvidia/container-images/driver/-/blob/master/ubuntu18.04/Dockerfile
#-------------------------------------------------
# ARG BASE_URL=https://us.download.nvidia.com/tesla
# ARG DRIVER_VERSION=450.80.02
# ENV DRIVER_VERSION=$DRIVER_VERSION

# RUN cd /tmp && \
#     curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run && \
#     sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run -x && \
#     cd NVIDIA-Linux-x86_64-$DRIVER_VERSION* && \
#     ./nvidia-installer --silent \
#                        --no-kernel-module \
#                        --install-compat32-libs \
#                        --no-nouveau-check \
#                        --no-nvidia-modprobe \
#                        --no-rpms \
#                        --no-backup \
#                        --no-check-for-alternate-installs \
#                        --no-libglx-indirect \
#                        --no-install-libglvnd \
#                        --x-prefix=/tmp/null \
#                        --x-module-path=/tmp/null \
#                        --x-library-path=/tmp/null \
#                        --x-sysconfig-path=/tmp/null && \
#     rm -rf /tmp/*


# Intel(VAAPI)のドライバをインストールする例
#-------------------------------------------------
# RUN apt-get update -q && apt-get install -y \
#     i965-va-driver \
#     && rm -rf /var/lib/apt/lists/*




#-------------------------------------------------------------------------------------------------
# 開発PCでのみ必要なツールのインストールや追加設定など
# ※シミュレーション対戦環境/実機環境でも必要な処理はdocker/core/Dockerfileに記載して下さい
#-------------------------------------------------------------------------------------------------
#  RUN apt-get update -q && apt-get install -y --no-install-recommends \
#      x11-apps \
#      && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=newt
