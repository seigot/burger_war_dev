ARG CORE_VERSION=latest
FROM burger-war-core:$CORE_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# 情報調査用のコマンドをインストール
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        lshw \
        kmod \
        lsof \
    && rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------------------------
# VNC/デスクトップ環境のインストール
#-------------------------------------------------------------------------------------------------
# tiniのインストール(PID1問題回避用)
# 参考: https://text.superbrothers.dev/200328-how-to-avoid-pid-1-problem-in-kubernetes/
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# 必要なツールライブラリのインストール
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        software-properties-common
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        supervisor \
        net-tools \
        zenity \
        xz-utils \
        dbus-x11 \
        x11-utils \
        alsa-utils \
        mesa-utils \
        libgl1-mesa-dri \
        xvfb \
        x11vnc \
    && rm -rf /var/lib/apt/lists/*

# デスクトップ環境のインストール
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        lxde \
        gtk2-engines-murrine \
        gnome-themes-standard \
        gtk2-engines-pixbuf \
        gtk2-engines-murrine \
        arc-theme \
        ttf-ubuntu-font-family \
    && rm -rf /var/lib/apt/lists/*

# 日本語化パッケージ
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        language-pack-ja-base \
        language-pack-ja \
        fonts-takao \
        ibus-mozc \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && fc-cache -f

#-------------------------------------------------------------------------------------------------
# コンテナ内で使いたいツールなどをインストール
#-------------------------------------------------------------------------------------------------
RUN apt-get update -q && apt-get install -y --no-install-recommends \
        firefox \
    && rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------------------------
# 環境設定
#-------------------------------------------------------------------------------------------------
# gnome-terminalが起動しない対策
RUN echo "export LANG=ja_JP.UTF-8" >> /home/developer/.bashrc \
    && echo "export LANGUAGE=ja_JP:ja" >> /home/developer/.bashrc \
    && echo "export LC_ALL=ja_JP.UTF-8" >> /home/developer/.bashrc \
    && echo "export GTK_IM_MODULE=ibus" >> /home/developer/.bashrc \
    && echo "export QT_IM_MODULE=ibus" >> /home/developer/.bashrc \
    && echo "export XMODIFIERS='@im=ibus'" >> /home/developer/.bashrc 

# gnome-terminal起動時の以下の警告対策
# Couldn't register with accessibility busµ
RUN echo "NO_AT_BRIDGE=1" >> /home/developer/.bashrc

# 必要なファイルのコピー
COPY vnc/rootfs /
RUN chown developer:developer -R /home/developer
RUN chown developer:developer /entrypoint_vnc.sh \
    && chmod +x /entrypoint_vnc.sh

ENTRYPOINT ["/entrypoint_vnc.sh"]

ENV DEBIAN_FRONTEND=newt
