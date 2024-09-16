
#!/bin/zsh
cd /Users/emilyqin/Desktop/traffic-predictor
source /Users/emilyqin/.virtualenvs/cv/bin/activate
env > /Users/emilyqin/Desktop/traffic-predictor/env_variables.log
/Users/emilyqin/.virtualenvs/cv/bin/python /Users/emilyqin/Desktop/traffic-predictor/main.py >> /Users/emilyqin/Desktop/traffic-predictor/logfile.log 2>> /Users/emilyqin/Desktop/traffic-predictor/error.log
