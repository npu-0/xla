#!/bin/bash
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export XLA_FLAGS="--xla_dump_to=/tmp/dir_name"
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=100
export XLA_SAVE_HLO_FILE="/tmp/dir_name/HLO"
export XLA_DUMP_HLO_GRAPH=1
