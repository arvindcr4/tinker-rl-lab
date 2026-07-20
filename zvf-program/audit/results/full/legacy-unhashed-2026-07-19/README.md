# Legacy unhashed E1 evidence preservation

These records and manifests are byte-for-byte copies preserved before the
2026-07-19 exact-checkpoint evaluation repair. The held-out rows are
contiguous, but the listed prefixes predate mandatory
`completion_sha256` evidence. They are not accepted by the hardened campaign
or aggregate validators.

| Unit | Unhashed rows | Prior HF commit | Record SHA-256 | Manifest SHA-256 |
|---|---:|---|---|---|
| `grpo-seed-11` | 500 | `a499726aedf6144b637cf77943964a45704fb931` | `c5d86469d44c9a9f1d47ad147f02fd9a65feba1dfaac0945cba1b0103a2ae1a7` | `50d8ef8b0cf83eab931ad75e89c15da067888345d6c9c00f044fe0f54a89f102` |
| `grpo-seed-89` | 212 | `2a125e30189a2481dd1750cb5bbd2071ab3b5cd5` | `8c6b8806fc11bd458a90f4494e411fc58915a3a75d61bbb5a5268bc44cc81c6f` | `6fdc900f12d570dd09482c67d1bd2d195338b590a633d50979e74a468c0cd71b` |
| `grpo-seed-107` | 244 | `99ca971725c694340ad475e9c03d7248a21a46eb` | `e1d19a91f52221ce4f624e71c7df46dfa6bcae84c95365241c7dedf960401c0d` | `9eb2b7ccb73405bc9c893891e31c7418dffaa156906ddc43e558b05d933719b3` |
| `dapo-seed-131` | 196 | `f34a67a1348a1f556e0f1ba78c2812cadf06e5ed` | `430e5de4e5996bfc3d22ee3e340a811c50575039585b4286302897e2d989e1e2` | `7ebf59149706ed800b258bbc6c360cf062f02da6f69b130355433f6e0766c415` |
| `gspo-seed-11` | 388 | `37a2793c2138940f1ece2950dd56df3e1cdf7ccc` | `0407d3c7c4abdcf90b6691e7464f8f3c69d7f38efa9aa7c256a4cee8aec96de0` | `d14ff870c6bab08fa4d0a65281e608f4fc4039317261ce73fd0c5b9c5e25bbda` |
| `gspo-seed-71` | 420 | `d3b74ccd365ba40f467c24e68b64bc597eb6746c` | `1b2497b6d3005bc98dda1620eb4e3684f66aa14fea2dcf156f3fe42a0dcbca8a` | `945aa29cb03533f37b13ca36c15a7f02afe1fce0aa6c72aaf51aad4bf289496a` |

The private Hub repositories retain these commits as immutable provenance.
Repair must use the original source request and the exact checkpoint-30 tree;
it may replay evaluation only and must record a rewind receipt.
