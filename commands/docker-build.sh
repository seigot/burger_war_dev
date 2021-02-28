#!/bin/bash
###############################################################################
#-burger-war-core/burger-war-devのDockerfileをビルドする
#-
#+[USAGE]
#+  $0 [-a BUILDオプション(core/dev)] [-c BUILDオプション(core)] [-d BUILDオプション(dev)] [-k kitイメージのバージョン] [-t BUILDターゲット][-v 作成イメージのバージョン] [-h]
#+
#-[OPTIONS]
#-  -a options    burger-war-core/burger-war-devの'docker build'に追加で渡す引数を指定（複数回指定可能）
#-  -c options    burger-war-coreの'docker build'に追加で渡す引数を指定（複数回指定可能）
#-  -d options    burger-war-devの'docker build'に追加で渡す引数を指定（複数回指定可能）
#-  -k version    利用するburger-war-kitのバージョンを指定
#-  -t target     ビルドするターゲットの指定(dev|robo|sim|vnc) *coreは常にビルドされる
#-  -v version    'docker build -t'で指定するイメージのバージョンを指定 (default: latest)
#-  -h            このヘルプを表示
#-
###############################################################################
set -e
set -u
CMD_NAME=$(basename $0)

# 関数定義
#------------------------------------------------
usage_exit() {
  # ファイル冒頭のコメントからUSAGEを出力
  sed '/^[^#]/q' "$0"             \
  | sed -n '/^#+/s/^#+//p'        \
  | sed "s/\$0/${CMD_NAME}/g"     1>&2
  exit 1
}
help_exit() {
  # ファイル冒頭のコメントからヘルプを出力
  sed '/^[^#]/q' "$0"             \
  | sed -n '/^#[-+]/s/^#[-+]//p'  \
  | sed "s/\$0/${CMD_NAME}/g"     1>&2
  exit 0
}
print_error() {
  # 引数のエラーメッセージを出力
  echo -n -e "\e[31m"
  echo -e "$@" | xargs -I{} echo -e {}
  echo -n -e "\e[m"
}

# 設定値読み込み
#------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname $0)"; pwd)
source "${SCRIPT_DIR}/config.sh"

# Proxy設定
PROXY_OPTION=
[ -n "${HOST_HTTP_PROXY}"  ] && PROXY_OPTION="${PROXY_OPTION} --build-arg HTTP_PROXY=${HOST_HTTP_PROXY}"
[ -n "${HOST_HTTPS_PROXY}" ] && PROXY_OPTION="${PROXY_OPTION} --build-arg HTTPS_PROXY=${HOST_HTTPS_PROXY}"
[ -n "${HOST_FTP_PROXY}"   ] && PROXY_OPTION="${PROXY_OPTION} --build-arg FTP_PROXY=${HOST_FTP_PROXY}"
[ -n "${HOST_http_proxy}"  ] && PROXY_OPTION="${PROXY_OPTION} --build-arg http_proxy=${HOST_http_proxy}"
[ -n "${HOST_https_proxy}" ] && PROXY_OPTION="${PROXY_OPTION} --build-arg https_proxy=${HOST_https_proxy}"
[ -n "${HOST_ftp_proxy}"   ] && PROXY_OPTION="${PROXY_OPTION} --build-arg ftp_proxy=${HOST_ftp_proxy}"
[ -n "${PROXY_OPTION}" ] && PROXY_OPTION="${PROXY_OPTION} --build-arg no_proxy=127.0.0.1,localhost,${HOSTNAME} --build-arg NO_PROXY=127.0.0.1,localhost,${HOSTNAME}"

# オプション・引数解析
#------------------------------------------------
DEV_BUILD_OPTION=
CORE_BUILD_OPTION=
IMAGE_VERSION=latest
BUILD_TARGET=dev
BUILD_DOCKER_IMAGE_NAME=${DOCKER_IMAGE_PREFIX}-${BUILD_TARGET}
while getopts a:c:d:k:t:v:h OPT
do
  case $OPT in
    a  ) # burger-war-core/burger-war-devのdocker buildへの追加オプション引数指定
      CORE_BUILD_OPTION="${CORE_BUILD_OPTION} ${OPTARG}"
      DEV_BUILD_OPTION="${DEV_BUILD_OPTION} ${OPTARG}"
      ;;
    c  ) # burger-war-coreのdocker buildへの追加オプション引数指定
      CORE_BUILD_OPTION="${CORE_BUILD_OPTION} ${OPTARG}"
      ;;
    d  ) # burger-war-devのdocker buildへの追加オプション引数指定
      DEV_BUILD_OPTION="${DEV_BUILD_OPTION} ${OPTARG}"
      ;;
    k  ) # kitのイメージバージョンを指定してビルド
      KIT_VERSION="${OPTARG}"
      ;;
    t  ) # ビルドするターゲットを指定
      BUILD_TARGET="${OPTARG}"
      if [ -f "${DOCKER_ROOT_DIR}/${BUILD_TARGET}/Dockerfile" ]; then
        BUILD_DOCKER_IMAGE_NAME=${DOCKER_IMAGE_PREFIX}-${BUILD_TARGET}
      else
        print_error \
          "指定ターゲットのDockerfileは存在しません: ${DOCKER_ROOT_DIR}/${BUILD_TARGET}/Dockerfile\n" \
          "ターゲットには次のいずれかを指定して下さい: core, dev, robo, sim, vnc"
        exit 1
      fi
      ;;
    v  ) # Dockerイメージのバージョン指定
      IMAGE_VERSION="${OPTARG}"
      ;;
    h  ) # ヘルプの表示
      help_exit
      ;;
    \? ) # 不正オプション時のUSAGE表示
      usage_exit
      ;;
  esac
done
shift $((OPTIND - 1))

# burger-war-kitのイメージを取得
#------------------------------------------------
set -x
docker pull ghcr.io/p-robotics-hub/burger-war-kit:${KIT_VERSION}
set +x

# コアイメージ用のDockerfileのビルド
#------------------------------------------------
set -x
docker build \
  ${CORE_BUILD_OPTION} \
  --build-arg KIT_VERSION=${KIT_VERSION} \
  ${PROXY_OPTION} \
  -f "${CORE_DOCKER_FILE_PATH}" \
  -t ${CORE_DOCKER_IMAGE_NAME}:${IMAGE_VERSION} \
  "${DOCKER_ROOT_DIR}"
set +x

# 開発環境用のDockerfileのビルド
#------------------------------------------------
set -x
docker build \
  ${DEV_BUILD_OPTION} \
  --build-arg CORE_VERSION=${IMAGE_VERSION} \
  ${PROXY_OPTION} \
  -f "${DOCKER_ROOT_DIR}/${BUILD_TARGET}/Dockerfile" \
  -t ${BUILD_DOCKER_IMAGE_NAME}:${IMAGE_VERSION} \
  "${DOCKER_ROOT_DIR}"
set +x

cat <<-EOM
#--------------------------------------------------------------------
# 以下のイメージを作成しました
# カスタマイズ用: ${CORE_DOCKER_IMAGE_NAME}:${IMAGE_VERSION}
# 開発環境用　　: ${BUILD_DOCKER_IMAGE_NAME}:${IMAGE_VERSION}
#--------------------------------------------------------------------
EOM
