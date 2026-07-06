# Dropped v2 candidates — floor-failure evidence (ADR-009 as amended)

The four jointly-trained partner-side candidates below FAILED the
committed competence floor (0.75, fingerprint probe vs the co-carry
reference ego) and were dropped from `cocarry_partners@v2` per the
set rule ("a member no ego can work with measures nothing"):

| candidate | floor success | pair checkpoint |
| --- | --- | --- |
| `joint_s0` | 0.000 | `local://artifacts/4ace772a2efe7dd3_step150000.pt` |
| `joint_s1` | 0.000 | `local://artifacts/24e5f7483ad7b86e_step200000.pt` |
| `joint_s2` | 0.050 | `local://artifacts/e2f99cc34a4c5356_step50000.pt` |
| `joint_s3` | 0.650 | `local://artifacts/461dbbcae360f85e_step150000.pt` |

`joint_s4` (floor success 0.800) is the one admitted learned member.
This is a measured cross-play finding, preserved as evidence: the
jointly-trained pairs co-adapted to THEIR egos (see the B-JOINT
leaderboard row, evaluated as-a-pair) and four of five cannot
cooperate with the scripted reference ego — the partner-familiarity
phenomenon the zoo exists to measure, caught by the committed floor
gate rather than asserted. Probe bundles in this directory are the
raw evidence (each was produced by the standard fingerprint probe;
they are NOT part of the v2 archive and carry no set membership).
