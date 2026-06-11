## File Directory

```text
./
│  PDEformer_inference.ipynb                     # English interactive notebook for inference examples
│  PDEformer_inference_CN.ipynb                  # Chinese interactive notebook for inference examples
│  pip-requirements.txt                          # Python dependency list
│  README.md                                     # English documentation
│  README_CN.md                                  # Chinese documentation
├─configs
│  └─inference                                   # Configurations for loading pretrained PDEformer models
│         model-L.yaml                           # Size-L model configuration
│         model-M.yaml                           # Size-M model configuration
│         model-S.yaml                           # Size-S model configuration
├─docs
│  │  FILE_TREE.md                               # This file
│  │  FILE_TREE_CN.md                            # Chinese file tree
│  └─images                                      # Images used in README and notebooks
├─scripts
│      run_ui.sh                                 # Start the GUI demonstration
└─src
    │  inference.py                              # PDEformer inference helpers
    ├─cell                                       # PDEformer model architecture
    │  │  basic_block.py                         # Shared neural-network blocks
    │  │  env.py                                 # Model constants and switches
    │  │  wrapper.py                             # PDEformer model factory and checkpoint loading
    │  └─pdeformer
    │      │  function_encoder.py                # Function encoder module
    │      │  pdeformer.py                       # PDEformer network architecture
    │      ├─graphormer                          # Graphormer encoder modules
    │      └─inr_with_hypernet                   # INR + HyperNet modules
    ├─data
    │  │   env.py                                # DAG constants and data precision settings
    │  │   pde_dag.py                            # PDE-to-DAG construction and graph tensor generation
    │  └─multi_pde                               # PDE term and boundary builders used by the UI
    │         boundary.py                        # Legacy boundary helper required by boundary v2
    │         boundary_v2.py                     # Boundary-condition DAG helpers
    │         boundary_v2_from_dict.py           # Dictionary interface for UI boundary terms
    │         terms.py                           # PDE term DAG and LaTeX helpers
    │         terms_from_dict.py                 # Dictionary interface for UI PDE terms
    ├─ui                                         # GUI utilities
    │      basic.py                              # UI base classes
    │      database.py                           # PDE term database components
    │      dcr.py                                # DCR equation GUI demo
    │      elements.py                           # UI elements
    │      pde_types.py                          # UI PDE solver wrappers
    │      utils.py                              # UI plotting and expression helpers
    │      widgets.py                            # UI widgets for PDE terms
    └─utils                                      # Inference and visualization utilities
           load_yaml.py                          # YAML configuration loader
           tools.py                              # Miscellaneous helper functions
           visual.py                             # Plotting and animation helpers
```
