#!/bin/bash
set -e

python ./fate_flow/fate_flow_server.py >  /dev/null 2>&1 &
python ./fml_agent/fml_agent.py
