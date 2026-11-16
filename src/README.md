# Source directory

Read the implementation in three groups:

1. **Harnessᴿ resource coordination:** [orion/capvm](orion/capvm/README.md).
2. **Reference integration and agent:** [orion/core](orion/core), [orion/main.py](orion/main.py), the context/memory/task modules they use, and their [integration guide](../examples/scientific_agent/README.md).
3. **Application and reproduction support:** [orion/tools](orion/tools), [kg_rag](kg_rag), [orion/benchmark](orion/benchmark), [release_paths.py](release_paths.py) and [release_verify.py](release_verify.py).

`orion` is the existing application namespace. Keeping it preserves source provenance and the tested imports. It does not make the whole reference agent the Harnessᴿ contribution. Some supporting modules are shared between groups; the integration boundary is identified by files rather than by treating all of `orion/core` as resource management.
