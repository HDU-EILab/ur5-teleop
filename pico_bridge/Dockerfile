FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget git cmake build-essential \
    python3 python3-dev python3-pip python3-pybind11 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q \
    https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb \
    -O /tmp/xrobotoolkit.deb \
    && dpkg-deb -x /tmp/xrobotoolkit.deb / \
    && rm /tmp/xrobotoolkit.deb
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/xrt-build
COPY vendor/pybind /tmp/xrt-build/pybind
COPY vendor/sdk/include /tmp/xrt-build/pybind/include
COPY vendor/sdk/lib/libPXREARobotSDK.so /tmp/xrt-build/pybind/lib/libPXREARobotSDK.so
RUN cp pybind/lib/libPXREARobotSDK.so /usr/local/lib/ \
    && cd pybind \
    && python3 setup.py install \
    && ldconfig \
    && rm -rf /tmp/xrt-build

COPY probe.py /opt/pico/probe.py
COPY entrypoint.sh /opt/pico/entrypoint.sh
RUN chmod +x /opt/pico/entrypoint.sh

ENV LD_LIBRARY_PATH=/usr/local/lib
EXPOSE 63901
ENTRYPOINT ["/opt/pico/entrypoint.sh"]
