#!/usr/bin/bash

# shellcheck disable=SC2045

if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

if [[ -f "$MS_ASSET_DIR/data/mshab_checkpoints" ]]; then
    CKPT_DIR="$MS_ASSET_DIR/data/mshab_checkpoints"
else
    CKPT_DIR="mshab_checkpoints"
fi

for task in $(ls -1 "$CKPT_DIR/rl")
do
    for subtask in $(ls -1 "$CKPT_DIR/rl/$task")
    do
        if [[ $task == "set_table" ]]; then
            if [[ $subtask == "close" ]]; then
                continue
            fi
            if [[ $subtask == "open" ]]; then
                continue
            fi
        fi
        for obj_name in $(ls -1 "$CKPT_DIR/rl/$task/$subtask")
        do
            if [[ $obj_name == "all" ]]; then
                python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
            fi
        done

    done
done

# for task in $(ls -1 "$CKPT_DIR/rl")
# do
#     if [[ $task == "set_table" ]]; then
#         for subtask in $(ls -1 "$CKPT_DIR/rl/$task")
#         do
#             if [[ $subtask == "pick" ]]; then
#                 for obj_name in $(ls -1 "$CKPT_DIR/rl/$task/$subtask")
#                 do
#                     if [[ $obj_name == "all" ]]; then
#                         python -m mshab.utils.gen.gen_data "$task" "$subtask" "$obj_name"
#                     fi
#                 done
#             fi
#         done
#     fi
# done