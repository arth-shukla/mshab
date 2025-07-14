#!/usr/bin/bash

if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

TASKS=(
    tidy_house
    prepare_groceries
    set_table
)

for task in "${TASKS[@]}"
do
    python -m mshab.utils.gen.gen_data_sequential_task "$task"
done