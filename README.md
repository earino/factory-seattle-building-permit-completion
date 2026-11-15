---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Four-year final-inspection sign-off of newly issued Seattle building permits

Will a newly issued Seattle building permit for a substantive construction project reach final inspection sign-off within four years of issuance?

Derived from City of Seattle - Seattle Department of Construction and Inspections (SDCI), published on data.seattle.gov. This is not its official product.

Prediction time: The permit IssuedDate, when the permit is issued. Only fields written at submittal, during plan review, or at issuance are used; CompletedDate, StatusCurrent, ExpiresDate and post-issuance revisions are excluded from features. The label is observed retrospectively after the 48-month window.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.635545439690134,
    "brier": 0.17718219376270802
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.18889847635656057
  },
  "calendar_identity": {
    "auc": 0.5130716895355019,
    "brier": 0.2238072774370914
  }
}
```

## Limitations

- The outcome is an administrative record: a physically finished project whose final inspection was never logged, or whose permit lapsed, counts as non-completion, so the label measures closure of the permit as much as completion of the work.
- Current-vintage retrospectivity: EstProjectCost and HousingUnits are documented as subject to change if the project is modified, and the plan-review day columns are frequently null in 2016-era rows. Nothing here proves those fields were visible to an analyst in 2016; only that they are recorded on the current snapshot and are logically prior to issuance.
- The dataset is a live, daily-refreshed endpoint with no immutable archive; a rebuild cannot reproduce the byte-identical extract. Labels for eligible cohorts are nonetheless stable because every 48-month window closed before 2026-09-30, so later refreshes can only add completions that fall outside the window.
- The snapshot is a rolling window. Pre-2004 rows are sparse and 2004-2015 coverage is uneven, which is why the cohort window starts in 2016; older issuance dates in the file (including 1900-01-01 sentinels) must be filtered, not trusted.
- The City's Terms of Use bar commercial use of data that is or can be configured as a list of individuals. Contractor names, applicant-linked fields and street addresses are therefore dropped and only ZIP is kept; this is a preservation obligation on the release, not a formality.
- Single city and a single permitting authority, so external validity is limited to Seattle's permit regime; holdout cohorts for 2021-2022 also sit in a different construction-cost and post-pandemic regime than the training years, which is part of the task rather than an artefact to remove.
- EstProjectCost >= 50000 is used both for eligibility and as a feature, so the population definition partly depends on a mutable declared field; a small number of projects revised upward after issuance can enter the cohort retroactively.
- Deliberate exclusion of cross-row history means contractor track record, site-level phasing and prior-permit outcomes - the strongest real-world predictors - are absent; permits sharing a Development Site or ParentPermitNum remain correlated, so rows are not strictly independent across sibling permits of one project.
- ExpiresDate is mechanically IssuedDate plus 18 months in the sampled rows, so permit duration carries almost no signal and is not used as a feature.
- A same-day CompletedDate (issued and completed on the same date) occurs in the eligible slice, which suggests some permits are closed administratively on issue; such rows are kept because the documented field rule is unambiguous, but they depress the informativeness of the positive class.
- The outcome is an administrative final-inspection completion record, not occupancy or certificate of occupancy. A physically finished project whose final inspection was never logged counts as non-completion.
- A null CompletedDate is documented as generally still under inspection; the frozen task maps null to y=0. This is an evidenced field interpretation, not an assumption that an unknown outcome is negative after the 48-month window.
- The Socrata resource is a live daily-refreshed current snapshot with no immutable archive. Exact population counts are tied to the 2026-11-14 publisher evidence and may fail on a later refresh.
- There is no full-row historical availability ledger. Feature availability at issuance is assumed from field semantics and current-snapshot presence, not proven.
- Plan-review and housing-unit fields have substantial missingness. Missing numeric and token values are left blank, not filled with zero.
- EstProjectCost is mutable and is used both for population eligibility and as a feature, so a few projects may enter the cohort retrospectively after an upward revision.
- Rows may be correlated across sibling permits through ParentPermitNum or RelatedMUP; no site-level independence is claimed.
- The 2021-2022 holdout sits in a different construction-cost and post-pandemic regime than the training years. This is part of the task, not an artefact to remove.
- ExpiresDate is mechanically IssuedDate plus about 18 months and is not used as a feature.
- Same-day CompletedDate values occur; they are kept because the field rule is unambiguous, but they depress the informativeness of the positive class.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
