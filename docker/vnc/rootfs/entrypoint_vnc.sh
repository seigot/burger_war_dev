#!/bin/bash

DEVELOPER_NAME=developer
DEV_HOME=/home/${DEVELOPER_NAME}

# ユーザーの指定
sed -i -e "s|{{USER}}|${DEVELOPER_NAME}|" -e "s|{{HOME}}|${DEV_HOME}|" /etc/supervisor/conf.d/supervisord.conf

# x11vncへの追加引数
if [ -n "$X11VNC_ARGS" ]; then
    sed -i "s/^command=x11vnc.*/& ${X11VNC_ARGS}/" /etc/supervisor/conf.d/supervisord.conf
fi

# VNCパスワード設定
if [ -n "$PASSWORD" ]; then
    mkdir -p ${DEV_HOME}/.x11vnc
    echo -n "$PASSWORD" > ${DEV_HOME}/.x11vnc/password1
    x11vnc -storepasswd $(cat ${DEV_HOME}/.x11vnc/password1) ${DEV_HOME}/.x11vnc/password2
    chmod 400 ${DEV_HOME}/.x11vnc/password*
    sed -i "s!command=x11vnc.*!& -rfbauth ${DEV_HOME}/.x11vnc/password2!" /etc/supervisor/conf.d/supervisord.conf
    export PASSWORD=
fi

# VNC解像度設定
if [ -n "$RESOLUTION" ]; then
    sed -i "s/1024x768/$RESOLUTION/" /usr/local/bin/xvfb.sh
fi

# OpenBoxへの追加引数
if [ -n "$OPENBOX_ARGS" ]; then
    sed -i "s#^command=/usr/bin/openbox\$#& ${OPENBOX_ARGS}#" /etc/supervisor/conf.d/supervisord.conf
fi

# home folder
if [ ! -x "${DEV_HOME}/.config/pcmanfm/LXDE/" ]; then
    mkdir -p ${DEV_HOME}/.config/pcmanfm/LXDE/
    ln -sf /usr/local/share/lxde/desktop-items-0.conf ${DEV_HOME}/.config/pcmanfm/LXDE/
    chown -R ${DEVELOPER_NAME}:${DEVELOPER_NAME} ${DEV_HOME}
fi

if [ "$(id -u)" == "0" ]; then

  if [ "${HOST_USER_ID}" != "$(gosu ${DEVELOPER_NAME} id -u)" ]; then
    # ホストPCとUSER ID/GROUP IDを合わせる(ファイルアクセスできなくなる為)
    usermod -u ${HOST_USER_ID} -o -m -d /home/${DEVELOPER_NAME} ${DEVELOPER_NAME}
    groupmod -g ${HOST_GROUP_ID} ${DEVELOPER_NAME}
    chown -R ${DEVELOPER_NAME}:${DEVELOPER_NAME} /home/${DEVELOPER_NAME}
  else
    chown -R ${DEVELOPER_NAME}:${DEVELOPER_NAME} /home/${DEVELOPER_NAME}
  fi
  su - ${DEVELOPER_NAME}
fi

exec /bin/tini -- supervisord -n -c /etc/supervisor/supervisord.conf
