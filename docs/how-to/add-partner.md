# How-to: Add a new partner class

!!! note "Phase-0 placeholder"
    Full recipe added once `chamber.partners` is implemented in M4.

## Steps (sketch)

1. Subclass `PartnerBase` from `chamber.partners.interface`.
2. Implement the `FrozenPartner` protocol from `concerto.api.partner`.
3. Add a registry entry in `chamber.partners.__init__`.
4. Write a property test covering the `_FORBIDDEN_ATTRS` shield.

*(Full content added in M4.)*
