(_processing)=
# Processing
The main processing class is {py:class}`skretrieval.usarm.processing.USARMRetrieval` which handles the
orchestration of all of the retrieval elements.
Generally the class is constructed, and then the retrieval is executed by calling the `retrieve` method.
Changes to the orchestration procedure can be controlled by subclassing `USARMRetrieval` and
implementing several derived functions that are described here.

## Retrieval Phasing
Often it is desired to execute the retrieval in phases, i.e., retrieve a subset of the state vector
in one step, and then move on to the next step using the results from the first.  In `usarm`
we take the philosophy that the entire state vector and the measurement vector is defined in the
initialization step (constructing `USARMRetrieval`), and then that specific state vector elements and
measurement vector elements can be disabled/enabled when calling `retrieve`.
