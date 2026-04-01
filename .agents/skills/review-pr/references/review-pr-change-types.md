# PR Review: Domain & Signal Detection Reference

This file contains the canonical change-domain and signal detection tables for PR
review. Referenced by: `.agents/skills/review-pr/SKILL.md`

______________________________________________________________________

## Severity Mapping

- **CRITICAL**: use `deep` category
- **HIGH**: use `deep` category
- **MEDIUM**: use `unspecified-high` category
- **LOW**: use `quick` category

______________________________________________________________________

## L1 Domains and L2 Signals

## Domain 1: Distributed Runtime (CRITICAL/HIGH)

| L2 Signal       | File Path Pattern                                                                             | Code Pattern                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `process_group` | `areal/engine/fsdp_utils/`, `areal/engine/megatron_utils/`, `areal/experimental/engine/`      | `new_group`, `ProcessGroup`, `dist.get_rank(`                                      |
| `fsdp_core`     | `areal/engine/fsdp_engine.py`, `areal/engine/fsdp_utils/`                                     | `FSDP`, `fully_shard`, `FullyShardedDataParallel`                                  |
| `megatron_core` | `areal/engine/megatron_engine.py`, `areal/engine/megatron_utils/`                             | `MegatronEngine`, `pipeline`, `micro-batch`                                        |
| `collectives`   | `areal/engine/`, `areal/infra/rpc/`                                                           | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `barrier` |
| `mesh_dtensor`  | `areal/experimental/models/archon/`, `areal/engine/fsdp_utils/`                               | `DeviceMesh`, `DTensor`, `Shard(`, `Replicate(`, `distribute_tensor`               |
| `weight_sync`   | `areal/experimental/engine/archon_weight_sync.py`, `areal/api/engine_api.py`, `areal/engine/` | `WeightUpdateMeta`, `set_version`, `update_weights`                                |

## Domain 2: Model Compute & Attention (HIGH/MEDIUM)

| L2 Signal              | File Path Pattern                                                        | Code Pattern                                     |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------------------------ |
| `tree_attn`            | `areal/models/tree_attn/`                                                | `TreeAttention`, `tree_attn`, `TreeNode`, `tree` |
| `sdpa_varlen`          | `attention/sdpa.py`, `attention/varlen.py`, `areal/models/tree_attn/`    | `sdpa`, `flash_attn`, `varlen`, `causal_mask`    |
| `sp_cp_attention_mask` | `areal/models/tree_attn/`, `areal/experimental/models/archon/attention/` | `SequenceParallel`, `context_parallel`, `mask`   |
| `triton_kernel`        | `areal/models/tree_attn/triton_kernel.py`                                | `triton`, `kernel`, `autotune`                   |

## Domain 3: Inference Backend & Serving (HIGH)

| L2 Signal           | File Path Pattern                        | Code Pattern                                                     |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------- |
| `vllm_ext`          | `areal/engine/vllm_ext/`                 | `areal_vllm_server`, `vllm_worker_extension`, `pause_generation` |
| `vllm_remote`       | `areal/engine/vllm_remote.py`            | `vllm`, `OpenAI`, `request`                                      |
| `sglang_remote`     | `areal/engine/sglang_remote.py`          | `sglang`, `request`, `response`                                  |
| `request_lifecycle` | `areal/engine/`, `areal/infra/launcher/` | `enqueue`, `dequeue`, `cancel`, `timeout`                        |

## Domain 4: Service Orchestration (HIGH)

| L2 Signal                     | File Path Pattern                                                                                      | Code Pattern                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| `agent_service_routing`       | `areal/experimental/agent_service/gateway/`, `areal/experimental/agent_service/router/`                | `route`, `gateway`, `router`              |
| `inference_service_dataproxy` | `areal/experimental/inference_service/data_proxy/`, `areal/experimental/inference_service/controller/` | `DataProxy`, `controller`, `batch`        |
| `session_consistency`         | `areal/experimental/agent_service/`, `areal/experimental/inference_service/`                           | `session`, `affinity`, `history`, `state` |

## Domain 5: Workflow & Trainer Contract (HIGH/MEDIUM)

| L2 Signal                  | File Path Pattern                                                                               | Code Pattern                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `workflow_engine_boundary` | `areal/workflow/`, `areal/trainer/`, `areal/engine/`                                            | `RolloutWorkflow`, `arun_episode`, `agenerate`      |
| `dataset_surface`          | `areal/dataset/`                                                                                | `DataLoader`, `IterableDataset`, `get_*_dataset`    |
| `async_contract`           | `areal/workflow/`, `areal/experimental/agent_service/`, `areal/experimental/inference_service/` | `async def`, `await`, `aiofiles`, `asyncio`         |
| `weight_version_contract`  | `areal/api/engine_api.py`, `areal/workflow/`, `areal/trainer/`                                  | `WeightUpdateMeta`, `set_version`, `weight version` |

## Domain 6: API & Config Compatibility (MEDIUM)

| L2 Signal          | File Path Pattern                     | Code Pattern                            |
| ------------------ | ------------------------------------- | --------------------------------------- |
| `dataclass_schema` | `areal/api/`                          | `@dataclass`, `field(`, `__post_init__` |
| `cli_compat`       | `areal/api/cli_args.py`               | `Literal`, `help`, `default`            |
| `backward_compat`  | `areal/api/`, `areal/infra/launcher/` | `deprecated`, `compat`, `version`       |

## Domain 7: Numerics & Tensor Semantics (MEDIUM)

| L2 Signal             | File Path Pattern                                                       | Code Pattern                                          |
| --------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `shape_dtype`         | `areal/engine/`, `areal/models/`, `areal/trainer/`                      | `.view(`, `.reshape(`, `dtype=`, `.contiguous(`       |
| `numerical_stability` | `areal/engine/`, `areal/reward/`, `areal/utils/functional/`             | `log(`, `softmax`, `eps=`, `.clamp(`, `nan`, `inf`    |
| `reward_surface`      | `areal/reward/`                                                         | `reward_fn`, `AsyncRewardWrapper`, `MathVerifyWorker` |
| `mixed_precision_fp8` | `areal/engine/megatron_utils/fp8/`, `areal/experimental/models/archon/` | `fp8`, `bf16`, `fp16`, `mixed precision`              |

## Domain 8: Checkpoint & Recovery (CRITICAL/HIGH)

| L2 Signal         | File Path Pattern                                                   | Code Pattern                                    |
| ----------------- | ------------------------------------------------------------------- | ----------------------------------------------- |
| `dcp_consistency` | `areal/utils/async_checkpoint.py`, `areal/engine/**/checkpoint*.py` | `dcp.save`, `dcp.load`, `DistributedCheckpoint` |
| `optimizer_state` | `areal/engine/fsdp_utils/checkpoint.py`, `areal/utils/saver.py`     | `optimizer state`, `state_dict`                 |
| `resume_compat`   | `areal/utils/recover.py`, `areal/utils/saver.py`                    | `resume`, `load_state_dict`, `migration`        |

## Domain 9: Launcher & Infrastructure (HIGH/MEDIUM)

| L2 Signal                 | File Path Pattern                                                      | Code Pattern                                   |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| `launcher_resource_match` | `areal/infra/launcher/`                                                | `LaunchConfig`, `RayLauncher`, `SlurmLauncher` |
| `scheduler_contract`      | `areal/infra/scheduler/`, `areal/scheduler/`                           | `Scheduler`, `placement`, `resource`           |
| `rpc_transport`           | `areal/infra/rpc/`, `areal/experimental/inference_service/data_proxy/` | `RTensor`, `serialize`, `rpc`, `fetch`         |

## Domain 10: Low-Risk Hygiene (LOW)

| L2 Signal                 | File Path Pattern                                       | Code Pattern                                                      |
| ------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `tests_docs_config`       | `tests/`, `docs/`, `*.md`, `*.yaml`, `*.json`, `*.toml` | -                                                                 |
| `logging_import_security` | `areal/`, `examples/`                                   | `getLogger`, `print(`, `import *`, `api_key`, `token`, `password` |

______________________________________________________________________

## Must-Not-Regress Core Coverage

The refactor must preserve these existing review surfaces:

- Archon core: `areal/experimental/models/archon/`,
  `areal/experimental/engine/archon_engine.py`
- FSDP core: `areal/engine/fsdp_utils/`, `areal/engine/fsdp_engine.py`
- Megatron core: `areal/engine/megatron_engine.py`, `areal/engine/megatron_utils/`
- Reward: `areal/reward/`
- Dataset: `areal/dataset/`

______________________________________________________________________

## Cross-Domain Linkage Rules

| Detected Signal                                          | Auto-Linked Review                                  |
| -------------------------------------------------------- | --------------------------------------------------- |
| `tree_attn`                                              | Numerics & Tensor Semantics checks                  |
| `vllm_ext`                                               | Launcher & Infrastructure checks                    |
| `agent_service_routing` or `inference_service_dataproxy` | Workflow & Trainer async-contract checks            |
| `weight_sync`                                            | DTensor/process-group/checkpoint interaction checks |
| `rpc_transport`                                          | Distributed Runtime synchronization checks          |
| `mixed_precision_fp8` + Distributed Runtime              | mesh + weight-sync compatibility checks             |

______________________________________________________________________

## Risk Identification Guidance

### Distributed Runtime Risks

- Collective call order mismatch across ranks
- Wrong process-group scope in rank-sensitive logic
- Mesh dimension mismatch and invalid DTensor placement
- Weight version drift between rollout and training workers

### Model Compute & Attention Risks

- Attention mask inconsistency under TP/SP/CP paths
- Tree attention index/routing mismatch
- Kernel assumptions violating dtype/shape invariants
- Sequence packing alignment errors

### Service Orchestration Risks

- Session affinity or history drift across gateway/router/data proxy
- Async message handling holes and dropped tasks
- Controller/worker lifecycle desynchronization

### Inference Backend & Serving Risks

- Request lifecycle inconsistencies (enqueue/cancel/timeout)
- Worker state transitions leaving requests stranded
- Backend extension hooks drifting from runtime expectations

### Workflow & Trainer Contract Risks

- Workflow-engine contract drift across async boundaries
- Weight version handshake mismatch between rollout and train
- Trainer lifecycle transition inconsistencies

### API & Config Compatibility Risks

- Breaking config/schema changes without migration path
- Dataclass or CLI default changes altering behavior silently
- Missing validation for newly introduced fields

### Numerics & Tensor Semantics Risks

- Silent shape/dtype mismatch under distributed paths
- Unstable numerical operations in loss/reward logic
- Mixed-precision interaction regressions

### Checkpoint & Recovery Risks

- Partial-rank checkpoint participation
- Incompatible state key evolution
- Resume path breaking optimizer/model synchronization

### Launcher & Infrastructure Risks

- Resource assignment mismatching parallel strategy assumptions
- RPC transport metadata loss (shape/dtype/device)
- Startup/shutdown ordering races across processes

### Low-Risk Hygiene Risks

- Docs/config drift from actual runtime behavior
- Logging or import hygiene regressions
- Sensitive data exposure in logs or config
