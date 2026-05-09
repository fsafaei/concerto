# How-to: Add a new partner class

This guide walks a Phase-1 contributor through adding a new partner class to
the CHAMBER zoo. The contract is fixed in Phase 0 — see
[ADR-009 §Decision][adr-009] — and the steps below are the mechanical
end-to-end recipe.

## 1. Subclass `PartnerBase`

Every partner inherits from
[`chamber.partners.interface.PartnerBase`][partnerbase], which:

- stores `spec` (the [`PartnerSpec`][partnerspec] the registry passed in);
- shields the `_FORBIDDEN_ATTRS` set so the AHT no-joint-training constraint
  is enforced at runtime even when callers reach for raw `torch` methods.

Implement `reset(*, seed)` and `act(obs, *, deterministic)` to satisfy the
[`FrozenPartner`][frozenpartner] Protocol — that is the entire surface.

## 2. Register with `@register_partner`

```python
from chamber.partners import PartnerBase, PartnerSpec, register_partner

@register_partner("my_new_partner")
class MyNewPartner(PartnerBase):
    def reset(self, *, seed: int | None = None) -> None:
        ...

    def act(self, obs, *, deterministic=True):
        ...
```

The decorator records the class against the string id; double-registration
raises `ValueError` so name collisions are caught at import time.

## 3. Build instances via `load_partner`

```python
from chamber.partners import PartnerSpec, load_partner

spec = PartnerSpec(
    class_name="my_new_partner",
    seed=0,
    checkpoint_step=None,
    weights_uri=None,
    extra={"uid": "fetch"},
)
partner = load_partner(spec)
```

The `partner_id` (a 16-hex hash of `class_name`, `seed`, `checkpoint_step`,
`weights_uri`) is the stable identity the M3 conformal filter reads from
`obs["meta"]["partner_id"]` to detect mid-episode partner swap
([ADR-006 risk #3][adr-006]; [ADR-004 §risk-mitigation #2][adr-004]).

## 4. Property-test the shield

Add a property test that `partner.train()` (and every other forbidden name)
raises `AttributeError` referencing ADR-009. The existing
`tests/property/test_no_train_allowed.py` covers `PartnerBase` directly; for
new shielded subclasses, mirror that pattern.

## 5. Read pose from the comm packet

Partners read proprio from `obs["agent"][uid]` and pose from
`obs["comm"]["pose"][uid]` — the [ADR-003 §Decision][adr-003] fixed-format
channel. Do **not** reach into env-specific keys; partners are env-agnostic
by design.

[adr-003]: https://github.com/concerto-org/concerto/blob/main/adr/ADR-003-comm-interface.md
[adr-004]: https://github.com/concerto-org/concerto/blob/main/adr/ADR-004-safety-filter.md
[adr-006]: https://github.com/concerto-org/concerto/blob/main/adr/ADR-006-partner-policy-assumptions.md
[adr-009]: https://github.com/concerto-org/concerto/blob/main/adr/ADR-009-partner-zoo.md
[partnerbase]: ../reference/api.md
[partnerspec]: ../reference/api.md
[frozenpartner]: ../reference/api.md
