# IR-based Fault localization with multi-relational network

### Run the experiment

---

Executing main.py.

The four aggregators are as follows:

1. aggregator.py aggregates the three relationships (interaction relationship, co-citation relationship and similarity relationship).
2. aggregator_RC.py only aggregates the interaction relationship between bug reports and source code files.
3. aggregator_RC_CC.py aggregates the interaction relationship and co-citation relationship.
4. aggregator_RC_RR.py aggregates the interaction relationship and similarity relationship.
5. aggregator_1_hop.py aggregates the one-hop neighbors information.

### Requirements

---

+ python 3.8.10
+ pandas==2.0.1
+ scipy==1.10.1
+ torch==2.0.1
+ networkx==3.1
+ numpy==1.24.3
+ tqdm==4.65.0