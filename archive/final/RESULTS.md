# Results — Skyline Geolocation in Khumbu

> Companion to `METHODOLOGY.md` (authoritative methodology). All numbers
> reproducible from `results/*.json`.

## 1. Headline claim

With **confidence gating** (wide-FOV ≥200° + cross-scorer consensus),
the system localizes **7/25 wide-FOV panoramas at 85.7% precision (<1 km,
median ≈40 m)** and abstains on the rest. Without gating, RRF fusion of
three complementary skyline scorers improves pinpoint (<100 m) accuracy
from 8.8% → **11.8%** overall (20% → **28%** on wide-FOV) over the baseline
matcher.

## 2. Method comparison (all N=68)

| Scorer | Median | <100 m | <500 m | <1 km | <5 km | <10 km |
|---|---|---|---|---|---|---|
| Baseline | 15.4 km | 8.8% | 11.8% | 11.8% | 16.2% | 32.4% |
| bp(2,8) | 18.3 km | 7.4% | 7.4% | 8.8% | 10.3% | 16.2% |
| bp(3,16) | 14.1 km | 8.8% | 10.3% | 10.3% | 19.1% | 29.4% |
| RRF fusion | 15.8 km | 11.8% | 13.2% | 13.2% | 14.7% | 23.5% |
| Oracle | 9.1 km | 13.2% | 14.7% | 14.7% | 30.9% | 57.4% |

## 3. Wide-FOV subset (≥200° coverage, N=25)

| Scorer | Median | <100 m | <1 km | <5 km | <10 km |
|---|---|---|---|---|---|
| Baseline | 15.3 km | 20.0% | 24.0% | 28.0% | 44.0% |
| bp(3,16) | 12.8 km | 24.0% | 28.0% | 36.0% | 44.0% |
| RRF fusion | 14.4 km | 28.0% | 28.0% | 32.0% | 40.0% |
| Oracle | 4.8 km | 32.0% | 32.0% | 52.0% | 68.0% |

## 4. Confidence-gated accuracy

| Gate | Accepted | Precision <100 m | Precision <1 km | Median |
|---|---|---|---|---|
| Consensus only (`rrf_votes ≥3`) | 10/68 | 60.0% | 70.0% | ~42 m |
| Wide-FOV + consensus | 7/25 | **85.7%** | **85.7%** | ~40 m |
| No consensus (rejected) | 58 | 3.4% | 3.4% | 16.9 km |

Rejected panos are overwhelmingly wrong matches — gating removes noise,
not signal.

## 5. Consensus-accepted panoramas

| Pano | FOV | Baseline | bp(2,8) | bp(3,16) | Fused err |
|---|---|---|---|---|---|
| 45TvC0DOQFASM… | 262° | <0.1 km | <0.1 km | <0.1 km | **11 m** |
| 1Xr_csMd0tcO1… | 262° | <0.1 km | 24.2 km | <0.1 km | **13 m** |
| 2X37DP_ZxmaRy… | 218° | <0.1 km | <0.1 km | <0.1 km | **32 m** |
| 1d3odopqB0Iq_… | 262° | 28.0 km | <0.1 km | <0.1 km | **33 m** |
| -yiHVpEf_kKTG9… | 262° | <0.1 km | 0.1 km | <0.1 km | **42 m** |
| 3j6nusgkXQ2Xr… | 262° | 0.1 km | 16.1 km | <0.1 km | **42 m** |
| 0dZapPlBpQequ… | 174° | 0.1 km | 17.6 km | 18.8 km | **101 m** |
| 2d-jydcZQqi4L… | 202° | 18.0 km | 21.6 km | 15.3 km | 16.6 km ✗ |
| 2MRNSLlMRk7Y7… | 135° | 25.7 km | 17.8 km | 17.8 km | 17.8 km ✗ |
| 3h0LkLhIhKGGZ… | 132° | 10.7 km | 12.4 km | 25.7 km | 25.8 km ✗ |

Notable recoveries vs baseline: `3sTMX` 21.5 km → **87 m** (bp2,8);
`08rD1iBo1xcv` 12.4 km → **2.7 km**; `-KVX8POwi4FAW` RRF 33.3 → **12.9 km**.

## 6. Figures

| File | Content |
|---|---|
| `figures/fig1_error_cdf.png` | Error CDFs: all panos + wide-FOV subset |
| `figures/fig2_confidence_gates.png` | Precision by confidence gate |
| `figures/fig3_baseline_vs_rrf.png` | Per-pano scatter, baseline vs fusion |
| `figures/fig4_fov_vs_error.png` | Horizon coverage vs achievable error |

## 7. Data provenance

* DB: 1,338,650 horizon profiles (HORAYZON × Copernicus GLO-30), stride=2 scan.
* Queries: 68 multi-crop GSV panoramas, Khumbu region.
* True-VP rank resolved via KD-tree nearest row (±250 m tolerance).

## 8. Per-sample results (all 68 panos, RRF fusion)

| Pano | FOV° | base km | bp28 km | bp316 km | rrf m | votes |
|---|---|---|---|---|---|---|
| 45TvC0DOQFASM7NjOc | 262 | 0.0 | 0.0 | 0.0 | 11 | 3 |
| 1Xr_csMd0tcO1RgZfF | 262 | 0.0 | 24.2 | 0.0 | 13 | 3 |
| 2X37DP_ZxmaRyIb3xM | 218 | 0.0 | 0.0 | 0.0 | 32 | 3 |
| 1d3odopqB0Iq_4UIp2 | 262 | 28.0 | 0.0 | 0.0 | 33 | 3 |
| -yiHVpEf_kKTG9YGJ- | 262 | 0.0 | 0.1 | 0.0 | 42 | 3 |
| 3j6nusgkXQ2Xr9uYZd | 262 | 0.1 | 16.1 | 0.0 | 42 | 3 |
| 3sTMX-DwF7lIAkO0b9 | 262 | 21.5 | 0.1 | 0.2 | 87 | 2 |
| 0SWYlSUa8TQf7RgTdG | 168 | 0.1 | 0.5 | 18.5 | 94 | 2 |
| 0dZapPlBpQequXJK7P | 174 | 0.1 | 17.6 | 18.8 | 101 | 3 |
| 3fL1aRPNWfYm4ZqDX3 | 236 | 20.4 | 20.5 | 18.7 | 3527 | 2 |
| 3deMQ4aB_kzqrqpKVA | 262 | 0.0 | 16.8 | 23.9 | 6096 | 2 |
| 3o0DRtbXxxsdh5j8OI | 118 | 20.9 | 21.9 | 15.5 | 6720 | 2 |
| 3DxQAz5h6s6GG_AFh6 | 88 | 14.6 | 17.1 | 24.2 | 6851 | 2 |
| -mxDWIU-ey4hSX_hT3 | 120 | 29.2 | 23.0 | 10.7 | 7052 | 2 |
| --WyciZkeyJi1pLXhE | 240 | 5.8 | 5.4 | 7.5 | 7498 | 2 |
| 1awvUDTBiEyQcSGrje | 147 | 9.6 | 10.0 | 23.4 | 9998 | 2 |
| 3q5g7dAR1GToAlafhX | 132 | 15.3 | 10.9 | 8.2 | 10843 | 2 |
| 37eGSjaM0MPYLAfe6Q | 158 | 13.1 | 11.6 | 11.4 | 11592 | 2 |
| 36nWXcdnZruTDWFBQy | 175 | 23.2 | 19.0 | 14.0 | 12036 | 2 |
| 3888loagYHnU_5Atej | 118 | 12.6 | 12.3 | 12.2 | 12216 | 2 |
| 3MZn9qOQ8ny2VUoUCm | 256 | 18.5 | 15.4 | 12.4 | 12409 | 2 |
| 12zKB1VcMp1tr5ra7Q | 126 | 31.3 | 29.2 | 13.8 | 12414 | 2 |
| 3i1LxyMaw495_9WgI9 | 175 | 1.9 | 12.7 | 10.9 | 12696 | 2 |
| 2Ju05lU44_kUo7SSz_ | 130 | 15.0 | 22.7 | 1.8 | 12750 | 2 |
| -KVX8POwi4FAW7tL7h | 232 | 33.3 | 26.5 | 23.1 | 12922 | 2 |
| 2L4xDYpBmIRI1kcnAu | 112 | 23.2 | 15.1 | 4.3 | 13443 | 2 |
| 0LkC4UdxA9kMF9Y7Va | 175 | 21.9 | 11.9 | 13.6 | 13579 | 2 |
| 2qnw_OyCk2smVeOZmM | 128 | 5.0 | 23.1 | 29.8 | 13640 | 2 |
| 3NoWAk1-Z5OXzNTT1t | 132 | 15.1 | 13.1 | 14.4 | 14390 | 2 |
| -OSi9xFi8tJLYOq94v | 209 | 23.9 | 14.4 | 14.4 | 14445 | 2 |
| 0DCmqu8iMNNq1z9WoV | 250 | 6.1 | 9.4 | 13.5 | 14614 | 2 |
| 1iMFHgvO3P-bIZd2OG | 194 | 25.1 | 29.0 | 15.8 | 15312 | 2 |
| 08rD1iBo1xcvxj9_FD | 230 | 12.4 | 2.7 | 12.8 | 15426 | 2 |
| 1gMIKhp4j8ZVAeB-Gr | 248 | 15.3 | 16.2 | 13.3 | 15590 | 2 |
| 2vT9jvB7SFdT0A6682 | 214 | 6.0 | 16.0 | 16.0 | 16046 | 2 |
| 2d-jydcZQqi4LqAfXT | 202 | 18.0 | 21.6 | 15.3 | 16575 | 3 |
| 2VlKKP-waMXSSviSuA | 262 | 27.8 | 20.2 | 15.5 | 16665 | 2 |
| 0dNbPhLvsLdvDgLDb5 | 175 | 12.5 | 8.1 | 4.4 | 17068 | 2 |
| -7QU4SC9xwXBB0iDnU | 226 | 16.8 | 22.8 | 6.4 | 17640 | 2 |
| 3bzk_lKR4n0fu50agw | 148 | 18.0 | 31.7 | 9.5 | 17663 | 2 |
| 2MRNSLlMRk7Y7yW-M_ | 135 | 25.7 | 17.8 | 17.8 | 17763 | 3 |
| 4Gr7HoRsFGnarSLhCZ | 175 | 25.5 | 14.2 | 4.9 | 18031 | 2 |
| 2HzhK-7eatfTywYAoE | 142 | 24.2 | 18.7 | 9.8 | 18099 | 2 |
| 1o52_3zQLII2eBTV2T | 172 | 9.2 | 25.6 | 18.6 | 18674 | 2 |
| 0rAALTCSyk4drAQ7IT | 175 | 7.2 | 11.5 | 21.4 | 18750 | 2 |
| 1rwHGo31OGSQZvbR4t | 132 | 23.8 | 23.5 | 22.8 | 18753 | 2 |
| 061cCYn21HxjaJ7h2m | 175 | 4.4 | 19.1 | 21.7 | 19053 | 2 |
| 3MxkFaFq8cUzckVk1O | 251 | 19.4 | 25.3 | 4.8 | 19299 | 2 |
| 2sLGh6IeqZl2MZ_izY | 121 | 9.4 | 22.4 | 9.0 | 19406 | 2 |
| 0f-CfId7jqkG7ZwCY5 | 102 | 28.8 | 21.8 | 8.8 | 21758 | 2 |
| -d1Esm5HrWN_cfnrUG | 204 | 17.4 | 15.3 | 22.1 | 22144 | 2 |
| 1CRu1nOrx2rhVJeW7x | 116 | 21.3 | 22.3 | 22.3 | 22323 | 2 |
| 2BnP6OXoIDlOxvWq9c | 63 | 22.4 | 21.4 | 22.3 | 22373 | 2 |
| 0P4J2Hq1vEnM_hPjy- | 165 | 17.8 | 25.2 | 22.4 | 22451 | 2 |
| 0fWl3F3YRjohazVC11 | 244 | 17.7 | 13.9 | 13.1 | 22982 | 2 |
| -AcKqsLPPPrJR9ki3F | 160 | 9.5 | 12.3 | 14.3 | 23218 | 2 |
| 2vKP409hA7GMVkT38R | 87 | 27.3 | 28.1 | 24.2 | 25316 | 2 |
| 3h0LkLhIhKGGZSfuzP | 132 | 10.7 | 12.4 | 25.7 | 25804 | 3 |
| 3rx6VVMqkSVK2bTpx9 | 262 | 5.8 | 26.1 | 4.2 | 26119 | 2 |
| 4AQK-ijaR3ZiCtgy_P | 194 | 15.4 | 19.9 | 26.2 | 26218 | 2 |
| 0YO4hevl0TlMaGnZPL | 174 | 12.8 | 17.9 | 10.3 | 26336 | 2 |
| 2npgWxjuBtzTU0P8-O | 176 | 28.2 | 22.9 | 11.3 | 26419 | 2 |
| 42SrfVF0xA-l1KuJ_r | 175 | 6.3 | 26.9 | 27.2 | 26981 | 2 |
| 4IPOvHN0JFchytzzbg | 175 | 28.6 | 22.2 | 20.5 | 27107 | 2 |
| 1ogFsuCuCDeRqSnRnw | 200 | 2.4 | 21.5 | 28.5 | 28514 | 2 |
| 27HKuTj44Pg5aowtmR | 108 | 11.0 | 30.6 | 32.2 | 30409 | 2 |
| 0288ELrvdvaNvmaL4T | 116 | 20.8 | 33.4 | 28.2 | 33357 | 2 |
| 0j4fexEfmglJpoz4pf | 154 | 21.2 | 22.9 | 34.8 | 34764 | 2 |
