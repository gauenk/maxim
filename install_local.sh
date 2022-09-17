#!/bin/bash
python -m pip instal pip --upgrade
python -m pip install -e git+file:"$(pwd)"/../data_hub#egg=data_hub
python -m pip install -e git+file:"$(pwd)"/../cache_io#egg=cache_io
python -m pip install -e git+file:"$(pwd)"/../dnls#egg=dnls
python -m pip install -e .
