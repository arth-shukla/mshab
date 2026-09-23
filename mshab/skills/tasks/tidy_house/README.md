# TidyHouse granularity experiment

TidyHouse is the recommended next experiment for Layer-1/2 granularity. The
downloaded official sequential plans contain five object transfers and 20
atomic subtasks per episode:

```text
(NavigateToObject -> Pick -> NavigateToDestination -> Place) x 5
```

The downloaded RL inventory contains `navigate/all`, specialized and `all`
Pick policies, and specialized and `all` Place policies for all nine object
categories. This permits the same specialized-to-generic fallback design used
by SetTable without inventing a new low-level capability.

## Controlled graph variants

Both variants must use the same episode, contracts, policy preference order,
candidate nodes, and nominal 20-node execution path. Only the ownership of
nodes by Layer-1 sub-goals changes.

### Coarse

Use one result-oriented sub-goal per transferred object:

```text
object_i_delivered := at(object_i, destination_i)
```

Its Layer-2 subgraph contains navigation to the object, Pick candidates,
navigation to the destination, and Place candidates. For a five-object episode
this produces 5 sub-goals and 5 skill subgraphs.

### Fine

Use four verifiable state transitions per object:

```text
object_i_reachable
    -> object_i_holding
    -> destination_i_reachable
    -> object_i_placed
```

The corresponding predicates are:

```text
reachable(object_i)
holding(object_i)
reachable(destination_i)
at(object_i, destination_i)
```

For a five-object episode this produces 20 sub-goals and 20 skill subgraphs.
Pick and Place subgraphs retain specialized and generic achievers; therefore
the number of candidate nodes is unchanged between variants.

## Required measurements

Record structural metrics (`subgoals`, `subgraphs`, cross edges, mean nodes per
subgraph), task success, completed sub-goals, simulator steps, redundant skill
calls, recovery success, and replanning span. Run at least nominal, primary
Pick failure, primary Place failure, object-drop, and partially satisfied
initial-state conditions.

The current sequential evaluator can compare nominal paths, but a valid
granularity result also needs a fact-aware controller that marks sub-goals from
live predicates and re-decides after each node. Otherwise coarse and fine are
only two serializations of the same fixed 20-subtask plan.

