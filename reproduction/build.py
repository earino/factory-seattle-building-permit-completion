import calendar
import pathlib
import urllib.parse

import numpy as np
import pandas as pd
from source_io import fetch

SOURCE_BASE = 'https://data.seattle.gov/resource/76t5-zqzr.csv'
SOURCE_COLUMNS = [
    'permitnum', 'permitclass', 'permitclassmapped', 'permittypedesc',
    'estprojectcost', 'housingunitsadded', 'housingunitsremoved',
    'housingunits', 'applieddate', 'issueddate', 'totaldaysplanreview',
    'numberreviewcycles', 'daysoutcorrections', 'originalzip',
    'description', 'dependentbuilding', 'standardplan', 'parentpermitnum',
    'relatedmup', 'completeddate', 'permittypemapped', 'statuscurrent'
]
ID = 'permitnum'
TIME = 'issueddate'
TARGET = 'completed_within_48_months'
FEATURES = [
    'permitclass', 'permitclassmapped', 'permittypedesc', 'estprojectcost',
    'housingunitsadded', 'housingunitsremoved', 'housingunits',
    'applied_to_issued_days', 'totaldaysplanreview', 'numberreviewcycles',
    'daysoutcorrections', 'issue_month_sin', 'issue_month_cos',
    'originalzip', 'description_length', 'description_stfi',
    'description_per_plan', 'description_apartment', 'description_unit',
    'description_tenant', 'description_demolition', 'dependentbuilding',
    'standardplan', 'parent_permit_flag', 'related_mup_flag'
]
OUTPUT_COLUMNS = [ID, TIME, TARGET] + FEATURES
TRAIN_START = pd.Timestamp('2016-01-01')
TRAIN_END = pd.Timestamp('2019-01-01')
EVAL_END = pd.Timestamp('2021-01-01')
HOLDOUT_END = pd.Timestamp('2022-10-01')
COST_FLOOR = 50000.0
HORIZON_MONTHS = 48


def _add_months(ts, months):
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    year = ts.year + (ts.month - 1 + months) // 12
    month = (ts.month - 1 + months) % 12 + 1
    day = min(ts.day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day)


def _to_number(series):
    text = series.astype('string').str.strip()
    text = text.replace({
        'true': '1', 'false': '0', 'True': '1', 'False': '0',
        'TRUE': '1', 'FALSE': '0'
    })
    return pd.to_numeric(text, errors='coerce')


def _fetch_frames(cache):
    frames = []
    paths = []
    offset = 0
    limit = 50000
    while True:
        params = {
            '$select': ','.join(SOURCE_COLUMNS),
            '$where': "issueddate >= '2016-01-01T00:00:00' AND issueddate < '2022-10-01T00:00:00' AND estprojectcost >= 50000",
            '$order': 'permitnum,issueddate',
            '$limit': str(limit),
            '$offset': str(offset)
        }
        url = SOURCE_BASE + '?' + urllib.parse.urlencode(params)
        path = fetch(url, cache, 'seattle_building_permits_%s.csv' % offset)
        local = pathlib.Path(path)
        if not local.exists() or local.stat().st_size == 0:
            raise AssertionError('Source page is empty. selected archive member: %s' % path)
        frame = pd.read_csv(
            local, dtype=str, keep_default_na=False,
            na_values=['', 'null', 'NULL', 'None'], encoding='utf-8-sig'
        )
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if frame.empty:
            break
        frames.append(frame)
        paths.append(str(local))
        if len(frame) < limit:
            break
        offset += limit
        if offset > 250000:
            raise AssertionError('Source pagination exceeded bound. selected archive member: %s' % paths)
    if not frames:
        raise AssertionError('No source rows returned. selected archive member: []')
    return frames, paths


def _require_fields(frame, required, paths):
    missing = [c for c in required if c not in frame.columns]
    if missing:
        message = (
            'Source field check failed. selected archive member: %s; '
            'observed headers: %s; missing names: %s'
        ) % (paths, list(frame.columns), missing)
        raise AssertionError(message[:4000])


def _add_check(checks, check, observed, expected, passed, kind=None):
    if not passed:
        message = 'Source check failed: %s; observed: %s; expected: %s' % (check, observed, expected)
        raise AssertionError(message[:4000])
    item = {'check': check, 'observed': observed, 'expected': expected, 'passed': True}
    if kind is not None:
        item['kind'] = kind
    checks.append(item)


def _sample_observed(frame, permitnum, paths):
    rows = frame[frame[ID].astype('string').str.strip() == permitnum]
    if rows.empty:
        raise AssertionError(
            'Source sample row missing. selected archive member: %s; permitnum: %s; observed headers: %s'
            % (paths, permitnum, list(frame.columns))
        )
    row = rows.iloc[0]
    return {
        'permitclass': str(row['permitclass']).strip(),
        'permittypedesc': str(row['permittypedesc']).strip(),
        'estprojectcost': float(row['estprojectcost']),
        'applieddate': str(row['applieddate_dt'].date()) if pd.notna(row['applieddate_dt']) else None,
        'issueddate': str(row['issueddate_dt'].date()) if pd.notna(row['issueddate_dt']) else None,
        'completeddate': str(row['completeddate_dt'].date()) if pd.notna(row['completeddate_dt']) else None
    }


def build(output: pathlib.Path, cache: pathlib.Path):
    checks = []
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    frames, paths = _fetch_frames(cache)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    _require_fields(raw, SOURCE_COLUMNS, paths)
    _add_check(checks, 'source_rows', int(len(raw)), 'at least one row from the Socrata resource', len(raw) > 0)
    _add_check(checks, 'source_fields', list(raw.columns), SOURCE_COLUMNS, set(SOURCE_COLUMNS).issubset(set(raw.columns)))
    for col in [
        'estprojectcost', 'housingunitsadded', 'housingunitsremoved',
        'housingunits', 'totaldaysplanreview', 'numberreviewcycles',
        'daysoutcorrections', 'dependentbuilding', 'standardplan'
    ]:
        raw[col] = _to_number(raw[col])
    for col in ['applieddate', 'issueddate', 'completeddate']:
        raw[col + '_dt'] = pd.to_datetime(raw[col], errors='coerce').dt.normalize()
    issued = raw['issueddate_dt']
    cost = raw['estprojectcost']
    date_cost = issued.notna() & (issued >= TRAIN_START) & (issued < HOLDOUT_END) & cost.notna() & (cost >= COST_FLOOR)
    candidate = raw.loc[date_cost].copy()
    type_text = candidate['permittypemapped'].astype('string').str.strip().str.casefold()
    is_building = type_text.eq('building').fillna(False)
    excluded_counts = candidate.loc[~is_building, 'permittypemapped'].fillna('(missing)').value_counts().to_dict()
    eligible = candidate.loc[is_building].copy()
    _add_check(
        checks, 'excluded_permit_type_codes',
        {str(k): int(v) for k, v in excluded_counts.items()},
        'non-Building codes may exist in the date/cost slice and are counted, then excluded by the frozen population',
        len(eligible) > 0
    )
    _add_check(checks, 'population_query_count', int(len(eligible)), 24750, len(eligible) == 24750)
    year_counts = eligible['issueddate_dt'].dt.year.value_counts().sort_index()
    observed_years = {str(int(k)): int(v) for k, v in year_counts.items()}
    expected_years = {'2016': 3841, '2017': 3779, '2018': 3669, '2019': 3981, '2020': 3154, '2021': 3460, '2022': 2866}
    _add_check(checks, 'permit_type_population_by_year', observed_years, expected_years, observed_years == expected_years)
    invalid_applied = eligible['applieddate_dt'].notna() & (eligible['issueddate_dt'] < eligible['applieddate_dt'])
    invalid_completed = eligible['completeddate_dt'].notna() & (eligible['completeddate_dt'] < eligible['issueddate_dt'])
    invalid_mask = invalid_applied | invalid_completed
    invalid_count = int(invalid_mask.sum())
    _add_check(
        checks, 'invalid_chronology_rows', invalid_count,
        'source construction recipe requires dropping CompletedDate before IssuedDate or IssuedDate before AppliedDate; count recorded',
        invalid_count >= 0
    )
    eligible = eligible.loc[~invalid_mask].copy()
    if eligible.empty:
        raise AssertionError('No eligible rows after population and chronology filters. selected archive member: %s' % paths)
    deadline = eligible['issueddate_dt'].map(lambda x: _add_months(x, HORIZON_MONTHS))
    eligible[TARGET] = ((eligible['completeddate_dt'].notna()) & (eligible['completeddate_dt'] <= deadline)).astype('int64')
    eligible['applied_to_issued_days'] = (eligible['issueddate_dt'] - eligible['applieddate_dt']).dt.days
    eligible['description_length'] = eligible['description'].str.len().astype('float')
    description = eligible['description']
    description_text = description.fillna('').astype(str)
    description_patterns = [
        ('description_stfi', r'\bstfi\b'),
        ('description_per_plan', r'\bper\s+plan\b'),
        ('description_apartment', r'\bapartment'),
        ('description_unit', r'\bunit\b'),
        ('description_tenant', r'\btenant'),
        ('description_demolition', r'\bdemolition')
    ]
    for col, pattern in description_patterns:
        values = description_text.str.contains(pattern, case=False, regex=True, na=False)
        output_values = pd.Series(pd.NA, index=eligible.index, dtype='Int64')
        present = description.notna()
        output_values.loc[present] = values.loc[present].astype('int64')
        eligible[col] = output_values
    month = eligible['issueddate_dt'].dt.month.astype(float)
    eligible['issue_month_sin'] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    eligible['issue_month_cos'] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    eligible['parent_permit_flag'] = eligible['parentpermitnum'].notna().astype('int64')
    eligible['related_mup_flag'] = eligible['relatedmup'].notna().astype('int64')
    train_mask = (eligible['issueddate_dt'] >= TRAIN_START) & (eligible['issueddate_dt'] < TRAIN_END)
    zip_text = eligible['originalzip'].astype('string').str.strip()
    zip_train = zip_text.loc[train_mask].dropna()
    zip_train = zip_train[zip_train.ne('')]
    top_zips = zip_train.value_counts().head(20).index.tolist()

    def _map_zip(value):
        if pd.isna(value):
            return 'missing'
        text = str(value).strip()
        if not text:
            return 'missing'
        return text if text in top_zips else 'other'

    eligible['originalzip'] = zip_text.map(_map_zip)

    def _split(value):
        if value < TRAIN_END:
            return 'train'
        if value < EVAL_END:
            return 'eval'
        if value < HOLDOUT_END:
            return 'holdout'
        return None

    eligible['_split'] = eligible['issueddate_dt'].map(_split)
    if eligible['_split'].isna().any():
        raise AssertionError('Unassigned split rows. selected archive member: %s' % paths)
    eligible[TIME] = eligible['issueddate_dt'].dt.strftime('%Y-%m-%d')
    _require_fields(eligible, OUTPUT_COLUMNS, paths)
    missing_features = [c for c in FEATURES if c not in eligible.columns]
    if missing_features:
        raise AssertionError(
            'Constructed feature field check failed. selected archive member: %s; observed headers: %s; missing names: %s'
            % (paths, list(eligible.columns), missing_features)
        )
    duplicates = int(eligible[ID].duplicated().sum())
    _add_check(checks, 'unique_permitnum', duplicates, 0, duplicates == 0)
    target_values = sorted([int(v) for v in eligible[TARGET].dropna().unique().tolist()])
    _add_check(
        checks, 'target_binary', target_values, [0, 1],
        set(target_values).issubset({0, 1}) and eligible[TARGET].notna().all()
    )
    null_positive = int(((eligible['completeddate_dt'].isna()) & (eligible[TARGET] == 1)).sum())
    _add_check(checks, 'null_completed_negative', null_positive, 0, null_positive == 0)
    obs1 = _sample_observed(eligible, '6071900-CN', paths)
    exp1 = {
        'permitclass': 'Multifamily', 'permittypedesc': 'Addition/Alteration',
        'estprojectcost': 119697.0, 'applieddate': '2005-12-20',
        'issueddate': '2016-01-12', 'completeddate': '2016-01-12'
    }
    pass1 = (
        obs1['permitclass'] == exp1['permitclass'] and
        obs1['permittypedesc'] == exp1['permittypedesc'] and
        abs(obs1['estprojectcost'] - exp1['estprojectcost']) < 0.01 and
        obs1['applieddate'] == exp1['applieddate'] and
        obs1['issueddate'] == exp1['issueddate'] and
        obs1['completeddate'] == exp1['completeddate']
    )
    _add_check(checks, 'source_sample_6071900', obs1, exp1, pass1)
    obs2 = _sample_observed(eligible, '6352649-CN', paths)
    exp2 = {
        'permitclass': 'Multifamily', 'permittypedesc': 'New',
        'estprojectcost': 807103.0, 'applieddate': '2013-12-26',
        'issueddate': '2016-01-07', 'completeddate': '2018-09-04'
    }
    pass2 = (
        obs2['permitclass'] == exp2['permitclass'] and
        obs2['permittypedesc'] == exp2['permittypedesc'] and
        abs(obs2['estprojectcost'] - exp2['estprojectcost']) < 0.01 and
        obs2['applieddate'] == exp2['applieddate'] and
        obs2['issueddate'] == exp2['issueddate'] and
        obs2['completeddate'] == exp2['completeddate']
    )
    _add_check(checks, 'source_sample_6352649', obs2, exp2, pass2)
    min_cost = float(eligible['estprojectcost'].min())
    _add_check(checks, 'unit_cost_floor', min_cost, 'at least 50000 nominal USD', min_cost >= COST_FLOOR, kind='unit')
    positives = eligible.loc[eligible[TARGET] == 1].copy()
    if positives.empty:
        raise AssertionError('No positive labels after construction. selected archive member: %s' % paths)
    positive_deadline = positives['issueddate_dt'].map(lambda x: _add_months(x, HORIZON_MONTHS))
    max_over = int((positives['completeddate_dt'] - positive_deadline).dt.days.max())
    _add_check(
        checks, 'unit_horizon_48_months', max_over,
        'at most 0 days beyond IssuedDate plus 48 calendar months for positive rows',
        max_over <= 0, kind='unit'
    )
    train_coverage = {c: int(eligible.loc[train_mask, c].notna().sum()) for c in FEATURES}
    empty_features = [c for c, n in train_coverage.items() if n == 0]
    _add_check(
        checks, 'train_feature_coverage', train_coverage,
        'every declared feature has at least one non-null training value',
        len(empty_features) == 0
    )
    leak = sorted(set(FEATURES).intersection({'completeddate', 'statuscurrent', 'expiresdate'}))
    _add_check(checks, 'no_outcome_feature_leak', leak, [], len(leak) == 0)
    public = output / 'public'
    private = output / 'private'
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    parts = {}
    part_paths = {
        'train': public / 'train.csv',
        'eval': public / 'eval.csv',
        'holdout': private / 'holdout.csv'
    }
    for name in ['train', 'eval', 'holdout']:
        part = eligible.loc[eligible['_split'] == name].copy()
        if part.empty:
            raise AssertionError('Empty split: %s. selected archive member: %s' % (name, paths))
        part = part.sort_values([TIME, ID], kind='mergesort').reset_index(drop=True)
        parts[name] = part
        part[OUTPUT_COLUMNS].to_csv(part_paths[name], index=False, lineterminator='\n')
        header = pd.read_csv(part_paths[name], nrows=0).columns.tolist()
        header_ok = header == OUTPUT_COLUMNS
        if not header_ok:
            missing_header = [c for c in OUTPUT_COLUMNS if c not in header]
            raise AssertionError(
                'Output field check failed. selected archive member: %s; observed headers: %s; missing names: %s'
                % (paths, header, missing_header)
            )
        _add_check(checks, 'output_headers_%s' % name, header, OUTPUT_COLUMNS, header_ok)
    split_counts = {name: int(len(part)) for name, part in parts.items()}
    _add_check(
        checks, 'split_counts', split_counts,
        'each split nonempty and sum equals eligible after chronology filters; publisher pre-exclusion sizes were 11289/7135/6326',
        sum(split_counts.values()) == len(eligible) and all(v > 0 for v in split_counts.values())
    )
    window_ok = True
    for name, part in parts.items():
        dates = part['issueddate_dt']
        if name == 'train':
            window_ok = window_ok and bool(((dates >= TRAIN_START) & (dates < TRAIN_END)).all())
        elif name == 'eval':
            window_ok = window_ok and bool(((dates >= TRAIN_END) & (dates < EVAL_END)).all())
        else:
            window_ok = window_ok and bool(((dates >= EVAL_END) & (dates < HOLDOUT_END)).all())
    _add_check(
        checks, 'split_window_integrity',
        {name: [str(part[TIME].min()), str(part[TIME].max())] for name, part in parts.items()},
        'train [2016-01-01, 2019-01-01), eval [2019-01-01, 2021-01-01), holdout [2021-01-01, 2022-10-01)',
        window_ok
    )
    rate = float(parts['train'][TARGET].mean())
    _add_check(
        checks, 'target_rate_sanity', rate,
        'between 0.50 and 0.95, consistent with completion being the majority class',
        0.50 <= rate <= 0.95
    )
    return checks
