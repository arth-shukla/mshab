## Here are the atomic skills
### PrepareGroceries 21:
rl.prepare_groceries.navigate.all

rl.prepare_groceries.pick.002_master_chef_can
rl.prepare_groceries.pick.003_cracker_box
rl.prepare_groceries.pick.004_sugar_box
rl.prepare_groceries.pick.005_tomato_soup_can
rl.prepare_groceries.pick.007_tuna_fish_can
rl.prepare_groceries.pick.008_pudding_box
rl.prepare_groceries.pick.009_gelatin_box
rl.prepare_groceries.pick.010_potted_meat_can
rl.prepare_groceries.pick.024_bowl
rl.prepare_groceries.pick.all

rl.prepare_groceries.place.002_master_chef_can
rl.prepare_groceries.place.003_cracker_box
rl.prepare_groceries.place.004_sugar_box
rl.prepare_groceries.place.005_tomato_soup_can
rl.prepare_groceries.place.007_tuna_fish_can
rl.prepare_groceries.place.008_pudding_box
rl.prepare_groceries.place.009_gelatin_box
rl.prepare_groceries.place.010_potted_meat_can
rl.prepare_groceries.place.024_bowl
rl.prepare_groceries.place.all

### SetTable 11:
rl.set_table.navigate.all

rl.set_table.pick.013_apple
rl.set_table.pick.024_bowl
rl.set_table.pick.all

rl.set_table.place.013_apple
rl.set_table.place.024_bowl
rl.set_table.place.all

rl.set_table.open.fridge
rl.set_table.open.kitchen_counter

rl.set_table.close.fridge
rl.set_table.close.kitchen_counter

### TidyHouse 21: 
rl.tidy_house.navigate.all

rl.tidy_house.pick.002_master_chef_can
rl.tidy_house.pick.003_cracker_box
rl.tidy_house.pick.004_sugar_box
rl.tidy_house.pick.005_tomato_soup_can
rl.tidy_house.pick.007_tuna_fish_can
rl.tidy_house.pick.008_pudding_box
rl.tidy_house.pick.009_gelatin_box
rl.tidy_house.pick.010_potted_meat_can
rl.tidy_house.pick.024_bowl
rl.tidy_house.pick.all

rl.tidy_house.place.002_master_chef_can
rl.tidy_house.place.003_cracker_box
rl.tidy_house.place.004_sugar_box
rl.tidy_house.place.005_tomato_soup_can
rl.tidy_house.place.007_tuna_fish_can
rl.tidy_house.place.008_pudding_box
rl.tidy_house.place.009_gelatin_box
rl.tidy_house.place.010_potted_meat_can
rl.tidy_house.place.024_bowl
rl.tidy_house.place.all

## Here are the contracts
| Contract | Inputs | Preconditions | Effects | Deletes |
|---|---|---|---|---|
| `NavigateContract` | `goal` | `present(goal)` | `reachable(goal)` | 旧的 `reachable(*)` |
| `PickContract` | `object` | `reachable(object)`, `gripper_empty()` | `holding(object)` | `gripper_empty()` |
| `PlaceContract` | `object`, `destination` | `holding(object)`, `reachable(destination)` | `at(object,destination)`, `gripper_empty()` | `holding(object)` |
| `OpenContract` | `container` | `reachable(container)`, `closed(container)` | `open(container)` | `closed(container)` |
| `CloseContract` | `container` | `reachable(container)`, `open(container)` | `closed(container)` | `open(container)` |