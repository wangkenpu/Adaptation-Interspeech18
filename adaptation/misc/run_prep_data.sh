#!/bin/bash

# Copyright 2018    Ke Wang

set -euo pipefail

bash misc/prep_adaptation_data.sh
bash misc/prep_diff_duration_adapt_data.sh
bash misc/prep_new_test.sh
