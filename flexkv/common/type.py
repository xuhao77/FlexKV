from dataclasses import dataclass, field
from typing import Optional, Protocol, TypeVar, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flexkv.common.block import SequenceMeta


class RadixNodeLike(Protocol):
    """Structural type shared by every radix-node flavor the cache layer moves
    around, currently the C++ ``CRadixNode``. The cache_engine layer only ever
    calls ``size()`` on a node, so that is all the protocol needs to require --
    which also lets a future non-C++ node flavor satisfy it without a shared
    base class."""

    def size(self) -> int: ...


# Each engine works with one concrete node type, consistent within an instance
# — hence a bounded TypeVar rather than a union.
NodeT = TypeVar("NodeT", bound=RadixNodeLike)


@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    # Mooncake-store only: main KV longest prefix (may exceed num_matched_blocks
    # when SWA joint hit is shorter). PUT uses this to skip existing KV keys.
    kv_matched_blocks: int = 0
    last_ready_node: Optional["RadixNodeLike"] = None
    last_node: Optional["RadixNodeLike"] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    block_node_ids: Optional[np.ndarray] = None
    matched_pos: Optional[str] = None
    matched_node_ids: Optional[np.ndarray] = None #TODO id or ids? should we allow one req match results on multiple nodes?
    insert_to_local_cpu_index: bool = True
    # ===== SWA (node-mounted) — passed through from CMatchResult so GET can
    # select the SWA source from the same forward match instead of re-walking
    # the tree. The deepest fully-matched ready node carrying a live SWA slot,
    # and the ready-prefix block count ending at it. None / 0 when no SWA on
    # the matched path.
    last_swa_node: Optional["RadixNodeLike"] = None
    swa_hit_blocks: int = 0

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1


class CacheEngineLike(Protocol[NodeT]):
    """Common surface of the cache engines — ``CacheEngineAccel`` and
    ``HierarchyLRCacheEngine``. They are duck-typed (no shared base class); this
    protocol declares only the methods *every* engine implements. Generic over
    the engine's node type so a single instance stays consistent.

    P2P-only surface (``start`` / ``match_local`` / ``match_all`` / ``local_index``)
    lives on ``HierarchyLRCacheEngine`` alone and is deliberately NOT part of this
    protocol; the config-guarded call sites reach it directly, and the type checker
    flagging those accesses is expected."""

    def reset(self) -> None: ...
    def match(self, sequence_meta: "SequenceMeta") -> MatchResultAccel: ...
    def insert(self,
               sequence_meta: "SequenceMeta",
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = ...,
               is_ready: bool = ...,
               match_result: Optional[MatchResultAccel] = ...) -> Optional[NodeT]: ...
    def take(self,
             num_required_blocks: int,
             protected_node: Optional[NodeT] = ...,
             strict: bool = ...) -> np.ndarray: ...
    def recycle(self, physical_blocks: np.ndarray) -> None: ...
    def set_ready(self, node: NodeT, ready: bool, ready_length: int) -> None: ...
    def unlock(self, node: NodeT) -> None: ...
    def lock_node(self, node: NodeT) -> None: ...
