#----------------------------------------------------------
# Repository config values
#----------------------------------------------------------
# 使用するburger-war-kitイメージのバージョン
KIT_VERSION=latest

# Dockerイメージ名(公開パッケージ名)
DOCKER_IMAGE_NAME=burger-war

# Dockerイメージ名(ローカルパッケージ名)
DOCKER_IMAGE_PREFIX=burger-war
CORE_DOCKER_IMAGE_NAME=${DOCKER_IMAGE_PREFIX}-core

#----------------------------------------------------------
# Local config values
#----------------------------------------------------------
# PROXY設定
HOST_http_proxy=${http_proxy:-}
HOST_https_proxy=${https_proxy:-}
HOST_HTTP_PROXY=${HTTP_PROXY:-}
HOST_HTTPS_PROXY=${HTTPS_PROXY:-}
HOST_ftp_proxy=${ftp_proxy:-}
HOST_FTP_PROXY=${FTP_PROXY:-}

# VNCサーバのポート番号
VNC_PORT=5900
# VNCログイン時のパスワード(空文字の場合はパスワードなし)
VNC_PASSWORD=${VNC_PASSWORD:-}
# VNC接続の解像度
VNC_RESOLUTION=1280x800
# VNCのx11vncへの
VNC_X11VNC_ARGS=
# VNCのOpenBoxへの引数
VNC_OPENBOX_ARGS=

# ワークスペースのrootディレクトリのパス
HOST_WS_DIR=${HOME}/catkin_ws
# コンテナ上のワークスペースディレクトリ
CONTAINER_WS_DIR=/home/developer/catkin_ws

# kit/devのrootディレクトリパス
BURGER_WAR_KIT_DIR=${HOST_WS_DIR}/src/burger_war_kit
BURGER_WAR_DEV_DIR=${HOST_WS_DIR}/src/burger_war_dev

# ビルドするDockerfileパス
DOCKER_ROOT_DIR=${BURGER_WAR_DEV_DIR}/docker
CORE_DOCKER_FILE_PATH=${DOCKER_ROOT_DIR}/core/Dockerfile
