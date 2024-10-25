ARG PARENT_IMAGE
FROM $PARENT_IMAGE as frozen_stage

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /nve_ws

# Set default paths for tools and runtimes
ENV COLCON_HOME=/nve_ws/.colcon
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
ENV SHELL=/bin/bash

# Stop Qt apps (e.g. RViz) from using the MIT-SHM X11 extension that can cause visual glitches
ENV QT_X11_NO_MITSHM=1

# Source ROS setup files automatically upon default/shell entry
RUN echo 'source "/opt/ros/$ROS_DISTRO/setup.bash"' >> /root/.bashrc \
    && echo 'if [ -e /nve_ws/install/setup.bash ]; then' >> /root/.bashrc \
    && echo '  source /nve_ws/install/setup.bash' >> /root/.bashrc \
    && echo 'fi' >> /root/.bashrc \
    && echo '#!/usr/bin/env bash' > /nve_ws_entrypoint.bash \
    && echo 'source "/opt/ros/$ROS_DISTRO/setup.bash"' >> /nve_ws_entrypoint.bash \
    && echo 'if [ -e /nve_ws/install/setup.bash ]; then' >> /nve_ws_entrypoint.bash \
    && echo '  source /nve_ws/install/setup.bash' >> /nve_ws_entrypoint.bash \
    && echo 'fi' >> /nve_ws_entrypoint.bash \
    && echo 'exec "$@"' >> /nve_ws_entrypoint.bash \
    && chmod +x /nve_ws_entrypoint.bash

ENTRYPOINT ["/nve_ws_entrypoint.bash"]
CMD ["/bin/bash"]

FROM frozen_stage as base_stage

COPY install_deps.bash install_deps.bash
RUN apt-get update \
    && ./install_deps.bash \
    && rm -rf /var/lib/apt/lists/*

FROM base_stage as prod_stage

COPY install/ install/
