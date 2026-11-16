# Data and model provenance

The bundled random forest is the fitted artifact used by the paper, not the upstream authors' neural network. Its source SHA-256 is cb60829a64e6ac4f1590b49a496b650f06a73563c0b002d3b1579e1de642e182. The model implementation and its feature definitions are included.

Public sodium-electrolyte measurements: Zenodo record 18429013, https://doi.org/10.5281/zenodo.18429013. The original downloaded archive, its metadata, the three sodium training CSV files, and the selected measurement rows are included. The associated study is https://doi.org/10.1039/D6DD00149A. Consult the upstream record and its license for attribution and redistribution terms. No author credentials, bulk PDF collection, or full local database directory is included.

BGE-M3 weights are an external model dependency, https://huggingface.co/BAAI/bge-m3. Set EMBEDDING_MODEL to your downloaded model location or the model identifier. The paper's reported GPU and software-budget numbers are not promised on different hardware.

Neo4j retrieval experiments additionally require the evidence graph/vector index and the source evidence collection. Saved results include the returned evidence and query outputs for auditing, but they are not a substitute for a database snapshot when rerunning the live retrieval service. This package does not claim a fresh end-to-end retrieval rerun without those resources.

Official MCP filesystem server: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem. The package and dependency lock are supplied; third-party node_modules are intentionally omitted. Install with npm ci in experiments/mcp_sessions/server. This experiment manages real stdio sessions over frozen evidence/report files; it is not an autonomous literature-agent experiment.
