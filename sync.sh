#!/bin/bash
# sync application/ to the AWS server
rsync -auv application/ 13.58.151.242:~/application/
