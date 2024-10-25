FROM nve_ml_ros:latest  
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
   && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID --groups video -m $USERNAME \
   # [Optional] Add sudo support for the non-root user
   && apt-get update && apt-get install -y sudo \
   && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
   && chmod 0440 /etc/sudoers.d/$USERNAME \
   && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc 


USER $USERNAME

ENV WROKSPACE_HOME=/nve_ws

# Adding aliases for easier life and running it
RUN echo " Adding aliases to bashrc" \
   && echo "alias snoe='source /opt/ros/noetic/setup.bash'"     >> /home/$USERNAME/.bashrc \
   && echo "alias sros='snoe && source  /nve_ws/install/setup.bash'" >> /home/$USERNAME/.bashrc \
   && echo "if [ -f /opt/ros/noetic/setup.bash ]; then source /opt/ros/noetic/setup.bash; fi" >> /home/$USERNAME/.bashrc \
   && echo "if [ -f /nve_ws/install/setup.bash ]; then source /nve_ws/install/setup.bash; fi" >> /home/ros/.bashrc 
