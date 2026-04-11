import pandas as pd
from polar_app.repository import ProcessedSessionRepository
from polar_app.rr_pipeline import RRCleaningParams, analyze_rr_artifacts
from polar_app.etape_2bis import run_etape_2bis, LABEL_2BIS_A, LABEL_2BIS_B
import os

DEFAULT_OUTPUT_DIR = os.path.join(r'C:\5eme\stage\V1_project', 'data')
repo = ProcessedSessionRepository(DEFAULT_OUTPUT_DIR)
sessions = repo.list_sessions()
sessions = sorted(sessions, key=lambda s: (str(getattr(s, 'started_at', '') or ''), s.session_id))
session = sessions[-1]
_, rr_frame, _ = repo.load_session_data(session.session_id)
params = RRCleaningParams()
base = analyze_rr_artifacts(rr_frame, params)
rr_brut = rr_frame['rr_interval_ms'].astype('float64').to_numpy()
labels_modifies, label_2bis = run_etape_2bis(rr_brut, base.analysis_frame['label'].astype(str).tolist(), params=None)
res = analyze_rr_artifacts(rr_frame, params, labels_override=labels_modifies)
analysis = res.analysis_frame.copy()
analysis['label_2bis'] = label_2bis
analysis['t_min'] = analysis['t_offset_ms'] / 60000.0
positive_rr = analysis['rr_interval_ms'].where(analysis['rr_interval_ms'] > 0)
analysis['rr_plot_ms'] = positive_rr.clip(upper=max(float(params.rr_max_ms) * 1.25, 1800.0))
analysis['line_group'] = analysis['rr_plot_ms'].isna().astype(int).cumsum()
raw_line = analysis.loc[analysis['rr_plot_ms'].notna()].copy()
print('session', session.session_id)
print('raw_line rows', len(raw_line))
print('2bis A', int((raw_line['label_2bis']==LABEL_2BIS_A).sum()), '2bis B', int((raw_line['label_2bis']==LABEL_2BIS_B).sum()))
counts = raw_line.groupby('line_group').size().sort_values(ascending=False)
print('groups total', len(counts), 'groups >1', int((counts>1).sum()), 'top10', counts.head(10).to_dict())
window = raw_line[(raw_line['t_min']>=102) & (raw_line['t_min']<=117)][['t_min','rr_interval_ms','rr_plot_ms','label','label_2bis','line_group']]
print(window.head(80).to_string(index=False))
