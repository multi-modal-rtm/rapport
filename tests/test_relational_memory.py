import torch

from rapport.models.relational_memory import RelationalEdgeMemory, pair_index

NODE_DIM = 8
UTT_DIM = 6


def test_pair_index_bijective_and_contiguous():
    for n in range(2, 10):
        pairs = [(i, j) for i in range(n) for j in range(n) if i < j]
        indices = [pair_index(i, j) for i, j in pairs]
        assert len(indices) == len(set(indices)), "pair_index must be injective (no collisions)"
        assert sorted(indices) == list(range(len(pairs))), "pair_index must fill [0, P) with no gaps"


def test_pair_index_symmetric():
    for i in range(6):
        for j in range(6):
            if i != j:
                assert pair_index(i, j) == pair_index(j, i)


def test_pair_index_rejects_self_pair():
    import pytest

    with pytest.raises(ValueError):
        pair_index(3, 3)


def _module():
    torch.manual_seed(0)
    return RelationalEdgeMemory(node_dim=NODE_DIM, utt_dim=UTT_DIM)


def test_incident_only_updates_leave_other_edges_unchanged():
    mem = _module()
    node_state = {0: torch.randn(NODE_DIM), 1: torch.randn(NODE_DIM), 2: torch.randn(NODE_DIM)}
    edge_state = {pair_index(1, 2): torch.randn(mem.EDGE_DIM)}
    untouched_before = edge_state[pair_index(1, 2)].clone()

    # speaker 0 utters, incident to {1, 2}; edge (1,2) is NOT incident to 0.
    mem.update_incident_edges(edge_state, s=0, others=[1, 2], u_e=torch.randn(UTT_DIM), h_s_prev=torch.randn(NODE_DIM), node_state=node_state)

    assert torch.equal(edge_state[pair_index(1, 2)], untouched_before), "non-incident edge must be untouched"
    assert pair_index(0, 1) in edge_state and pair_index(0, 2) in edge_state, "incident edges must be created"


def test_update_creates_edges_from_implicit_zero_init():
    mem = _module()
    node_state = {1: torch.randn(NODE_DIM)}
    edge_state: dict[int, torch.Tensor] = {}
    u_e = torch.randn(UTT_DIM)
    h_s_prev = torch.zeros(NODE_DIM)  # speaker 0 is new -> zero prev state, per spec convention

    mem.update_incident_edges(edge_state, s=0, others=[1], u_e=u_e, h_s_prev=h_s_prev, node_state=node_state)

    # Reproduce by hand: e_prev should have been treated as zero.
    expected = mem.edge_gru(torch.cat([u_e, h_s_prev, node_state[1]]).unsqueeze(0), torch.zeros(1, mem.EDGE_DIM)).squeeze(0)
    assert torch.allclose(edge_state[pair_index(0, 1)], expected)


def test_ordering_attend_reads_freshly_updated_edges_not_stale():
    """A4: the node update (attend) must read edge state "at t", i.e. AFTER
    update_incident_edges has run for this same step -- not the pre-update
    (t-1) edge value.
    """
    mem = _module()
    node_state = {0: torch.randn(NODE_DIM), 1: torch.randn(NODE_DIM)}
    stale_edge = torch.randn(mem.EDGE_DIM)
    edge_state = {pair_index(0, 1): stale_edge}

    mem.update_incident_edges(
        edge_state, s=0, others=[1], u_e=torch.randn(UTT_DIM), h_s_prev=node_state[0], node_state=node_state
    )
    fresh_edge = edge_state[pair_index(0, 1)]
    assert not torch.allclose(fresh_edge, stale_edge), "sanity: the update must actually change the edge"

    context_with_fresh = mem.attend(0, [1], node_state, edge_state, node_state[0])
    edge_state_stale_copy = dict(edge_state)
    edge_state_stale_copy[pair_index(0, 1)] = stale_edge
    context_with_stale = mem.attend(0, [1], node_state, edge_state_stale_copy, node_state[0])

    assert not torch.allclose(context_with_fresh, context_with_stale), (
        "attend() must be sensitive to which edge value (fresh vs stale) it's given -- "
        "confirms the caller-enforced ordering (update before attend) actually matters"
    )


def test_attend_returns_zero_for_isolated_speaker():
    mem = _module()
    h_s_prev = torch.randn(NODE_DIM)
    context = mem.attend(0, [], {}, {}, h_s_prev)
    assert torch.equal(context, torch.zeros(mem.EDGE_DIM))


def test_edge_mean_returns_zero_for_isolated_speaker():
    mem = _module()
    result = mem.edge_mean(0, [], {}, torch.randn(NODE_DIM))
    assert torch.equal(result, torch.zeros(mem.EDGE_DIM))


def test_edge_mean_averages_incident_edges_only():
    mem = _module()
    e01 = torch.randn(mem.EDGE_DIM)
    e02 = torch.randn(mem.EDGE_DIM)
    e12 = torch.randn(mem.EDGE_DIM)  # not incident to speaker 0
    edge_state = {pair_index(0, 1): e01, pair_index(0, 2): e02, pair_index(1, 2): e12}

    result = mem.edge_mean(0, [1, 2], edge_state, torch.randn(NODE_DIM))
    assert torch.allclose(result, (e01 + e02) / 2)


def test_dyadic_n2_single_edge_no_special_casing():
    """A8: N=2 (one edge) must run through the same code path with no errors."""
    mem = _module()
    node_state = {0: torch.randn(NODE_DIM)}
    edge_state: dict[int, torch.Tensor] = {}

    mem.update_incident_edges(edge_state, s=1, others=[0], u_e=torch.randn(UTT_DIM), h_s_prev=torch.zeros(NODE_DIM), node_state=node_state)
    assert len(edge_state) == 1
    context = mem.attend(1, [0], node_state, edge_state, torch.zeros(NODE_DIM))
    mean = mem.edge_mean(1, [0], edge_state, torch.zeros(NODE_DIM))
    assert context.shape == (mem.EDGE_DIM,)
    assert mean.shape == (mem.EDGE_DIM,)


def test_gradient_flows_to_all_relational_parameters():
    mem = _module()
    node_state = {0: torch.randn(NODE_DIM, requires_grad=True), 1: torch.randn(NODE_DIM, requires_grad=True)}
    edge_state: dict[int, torch.Tensor] = {}
    u_e = torch.randn(UTT_DIM)

    mem.update_incident_edges(edge_state, s=0, others=[1], u_e=u_e, h_s_prev=node_state[0], node_state=node_state)
    context = mem.attend(0, [1], node_state, edge_state, node_state[0])
    mean = mem.edge_mean(0, [1], edge_state, node_state[0])
    loss = context.sum() + mean.sum()
    loss.backward()

    for name, param in mem.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient at {name}"
