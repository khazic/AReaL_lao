# PR Review: Domain Templates Reference

This file contains canonical domain templates for PR review. Referenced by:
`.agents/skills/review-pr/SKILL.md`

______________________________________________________________________

## Template Selection Rules

1. Select templates by detected L1 domains and L2 signals.
1. Use at most one primary template per domain.
1. Always include **General Logic & Boundary** for non-doc/config-only PRs.
1. Apply cross-domain linkage checks from `review-pr-change-types.md`.

______________________________________________________________________

## Universal Template

### General Logic & Boundary

```
Applicable: Any non-doc/config-only change
Checklist:
- Boundary condition correctness (empty inputs, singleton, max-size)
- Conditional logic correctness (branch inversion, short-circuit mistakes)
- Error-path behavior (exceptions propagated with actionable context)
- Return-value consistency across code paths
- No newly introduced hidden behavior changes
```

______________________________________________________________________

## Domain 1 Template: Distributed Runtime Review \[deep\]

```
Applicable signals: process_group, collectives, mesh_dtensor, weight_sync
Checklist:
- Process-group creation/usage/cleanup is rank-consistent
- Collective operations are called by all required ranks in consistent order
- DeviceMesh dimensions and DTensor placements are correct for each path
- Local/global tensor conversion boundaries are explicit and correct
- Weight version propagation and update ordering are deterministic
- No debug-only barriers left in hot path
```

## Domain 2 Template: Model Compute & Attention Review \[deep\]

```
Applicable signals: tree_attn, sdpa_varlen, sp_cp_attention_mask, triton_kernel
Checklist:
- Attention mask semantics preserved under TP/SP/CP
- Tree attention index/order invariants are maintained
- Kernel assumptions on dtype/shape/contiguity are satisfied
- No silent behavior change in sequence packing/unpacking
- Tensor layouts remain compatible with downstream modules
```

## Domain 3 Template: Inference Backend & Serving Review \[deep\]

```
Applicable signals: vllm_ext, vllm_remote, sglang_remote, request_lifecycle
Checklist:
- Request lifecycle (enqueue, execution, cancellation, timeout) is coherent
- Worker state transitions are safe under concurrency
- Backend-specific extension points stay API-compatible
- Error handling does not strand in-flight requests
- Versioning/weight-update interactions are explicit and safe
```

## Domain 4 Template: Service Orchestration Review \[deep\]

```
Applicable signals: agent_service_routing, inference_service_dataproxy, session_consistency
Checklist:
- Gateway/router/data-proxy routing rules are deterministic
- Session affinity and history consistency are preserved
- Controller/worker coordination has no lost-update window
- Async boundaries avoid blocking operations in critical paths
- Failure/retry behavior does not duplicate or drop work
```

## Domain 5 Template: Workflow & Trainer Contract Review \[deep\]

```
Applicable signals: workflow_engine_boundary, async_contract, weight_version_contract
Checklist:
- RolloutWorkflow and Engine interfaces remain contract-compatible
- Async flow uses await consistently and avoids sync I/O in async paths
- Weight update/version handshake is preserved end-to-end
- Trainer lifecycle transitions are valid for all execution branches
- Call ordering assumptions across trainer/workflow/engine are unchanged or justified
```

## Domain 6 Template: API & Config Compatibility Review \[unspecified-high\]

```
Applicable signals: dataclass_schema, cli_compat, backward_compat
Checklist:
- Public API signature and default value changes are intentional and compatible
- Dataclass validation remains complete and informative
- CLI options preserve expected compatibility semantics
- New fields include safe defaults or explicit migration handling
- Breaking changes are documented and scoped
```

## Domain 7 Template: Numerics & Tensor Semantics Review \[unspecified-high\]

```
Applicable signals: shape_dtype, numerical_stability, mixed_precision_fp8
Checklist:
- Tensor shape/dtype transitions are explicit and internally consistent
- Numerical stability is protected (log/division/softmax/clamp paths)
- Mixed-precision behavior is correct for forward + backward + reduce paths
- In-place and view/reshape operations do not corrupt gradient flow
- Device placement and dtype combinations remain legal across code paths
```

## Domain 8 Template: Checkpoint & Recovery Review \[deep\]

```
Applicable signals: dcp_consistency, optimizer_state, resume_compat
Checklist:
- Save/load requires and enforces all-rank participation where needed
- State dict naming/structure is stable or migration-safe
- Optimizer state sharding/gather behavior is consistent
- Resume path restores model + optimizer + version state coherently
- Async checkpoint behavior preserves ordering and durability assumptions
```

## Domain 9 Template: Launcher & Infrastructure Review \[unspecified-high\]

```
Applicable signals: launcher_resource_match, scheduler_contract, rpc_transport
Checklist:
- Resource assignment matches declared parallel strategy assumptions
- Scheduler decisions preserve required placement/affinity constraints
- RPC serialization/deserialization keeps shape/dtype/device semantics
- Transport retries/timeouts do not violate idempotency expectations
- Cross-process startup/shutdown ordering is robust
```

## Domain 10 Template: Low-Risk Hygiene Review \[quick\]

```
Applicable signals: tests_docs_config, logging_import_security
Checklist:
- Tests/docs/config edits are internally consistent and non-misleading
- Logging follows project conventions and avoids sensitive leakage
- No wildcard imports or obvious dependency hygiene regressions
- No accidental secrets/keys/tokens introduced
```

______________________________________________________________________

## Signal-Specific Add-On Checklists

Use these only when corresponding L2 signals are detected.

### `tree_attn` Add-On \[deep\]

```
- Node/edge indexing is deterministic and shape-safe
- Tree traversal order matches attention mask semantics
- FSDP/Megatron/Archon variant modules remain behaviorally aligned
```

### `vllm_ext` Add-On \[deep\]

```
- Server and worker extension hooks still match upstream expectations
- Request pause/resume/cancel semantics remain coherent
- Integration-specific monkey-patches are scoped and guarded
```

### `agent_service_routing` / `inference_service_dataproxy` Add-On \[deep\]

```
- Route selection and fallback ordering are deterministic
- Data proxy transformations preserve payload integrity
- Session-key partitioning logic is collision-safe
```

### `weight_sync` Add-On \[deep\]

```
- Versioned updates are monotonic and race-safe
- Broadcast/all-gather points are aligned with consumer expectations
- Local caching behavior cannot serve stale weights indefinitely
```

### `rpc_transport` Add-On \[unspecified-high\]

```
- RTensor conversion is reversible and metadata-complete
- Batch fetch/request framing preserves ordering and boundaries
- Retry logic does not replay non-idempotent actions incorrectly
```
