ARG KIT_VERSION=latest
FROM ghcr.io/p-robotics-hub/burger-war-kit:$KIT_VERSION

#-------------------------------------------------------------------------------------------------
# シミュレーション対戦環境/実機環境での実行に必要なパッケージをインストールする
#-------------------------------------------------------------------------------------------------
# apt-getで必要なパッケージをインストールする例
#-------------------------------------------------
# RUN apt-get update -q && apt-get install -y --no-install-recommends \
#     ros-${ROS_DISTRO}-dwa-local-planner \
#     ros-${ROS_DISTRO}-global-planner \
#     libarmadillo-dev \
#     libarmadillo8 \
#     && rm -rf /var/lib/apt/lists/*


# pipで必要なパッケージをインストールする例
#-------------------------------------------------
# RUN yes | pip install \
#     transitions \
#     pygraphviz


# ホストで用意したファイルをインストールする例
#-------------------------------------------------
# COPY core/myexec /usr/local/bin/myexec
# RUN chmod +x /usr/local/bin/myexec
