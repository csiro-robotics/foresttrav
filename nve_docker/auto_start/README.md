# Autostart
To deploy the model automatically we deploy the model as a `systmd` service

## Setup:
- platform.env in the $HOME directory
  - Contains the environment variables 
- te_estimator.service in `/etc/systemd/system/`
  - calls the `../run_deployment/run_te_estimation.bash` script and sets the environment variables
