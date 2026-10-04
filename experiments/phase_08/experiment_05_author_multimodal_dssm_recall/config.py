"""Protocol is explicit, not inherited from our earlier concat experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parent
OUT = ROOT / 'results/phase_08/experiment_05_author_multimodal_dssm_recall'
COMMIT = '39b7767372e46dfa5cd20b689d1c31a6d608ea69'
MODELS = {
    'A0': {'features': 'F', 'images': True, 'id_dropout': 0.0},
    'A1': {'features': 'F', 'images': True, 'id_dropout': 0.5},
    'B0': {'features': 'S', 'images': True, 'id_dropout': 0.0},
    'B1': {'features': 'S', 'images': True, 'id_dropout': 0.5},
    'T0': {'features': 'S', 'images': False, 'id_dropout': 0.0},
    'T1': {'features': 'S', 'images': False, 'id_dropout': 0.5},
}
PROTOCOL = dict(train_samples=254583, valid_samples=47733,
                valid_requests=13594, items=1983938, history_n=20,
                output_dim=128, temperature=0.07, batch_size=768,
                lr=1e-3, weight_decay=1e-2, epochs=3, patience=2,
                proxy_requests=5000, proxy_candidates=100000,
                workers=0, loss='author_easy_unmasked_inbatch',
                selection='fixed_proxy_request_macro_Recall@500', test_opened=False)
H2 = ROOT / 'results/phase_07/experiment_04_h2_retrieval_dimension_ablation'
ITEM_BASIC = ['video_duration', 'video_height', 'video_width', 'image_num',
              'content_length', 'commercial_flag']
ITEM_COUNTS = ['imp_num', 'click_num', 'like_num', 'collect_num', 'comment_num',
               'share_num', 'accum_like_num', 'accum_collect_num', 'accum_comment_num',
               'view_time', 'valid_view_times', 'full_view_times', 'imp_rec_num',
               'click_rec_num', 'rec_like_num', 'rec_collect_num', 'rec_comment_num',
               'rec_share_num', 'rec_follow_num', 'rec_view_time']
# No aliases to unrelated fields. Rates are derived from precisely these inputs.
RATE_INPUTS = {
    'ctr': ('click_num', 'imp_num'), 'like_rate': ('like_num', 'click_num'),
    'collect_rate': ('collect_num', 'click_num'),
    'comment_rate': ('comment_num', 'click_num'), 'share_rate': ('share_num', 'click_num'),
    'valid_view_time_rate': ('valid_view_times', 'view_time'),
    'full_view_time_rate': ('full_view_times', 'view_time'),
    'rec_ctr': ('click_rec_num', 'imp_rec_num'),
    'rec_like_rate': ('rec_like_num', 'click_rec_num'),
    'rec_collect_rate': ('rec_collect_num', 'click_rec_num'),
    'rec_share_rate': ('rec_share_num', 'click_rec_num'),
    'rec_follow_rate': ('rec_follow_num', 'click_rec_num'),
    'rec_view_time_rate': ('rec_view_time', 'view_time'),
}
ITEM_NUMERIC = ITEM_BASIC + ITEM_COUNTS + list(RATE_INPUTS)
USER_NUMERIC = ['fans_num', 'follows_num'] + [f'dense_feat{i}' for i in range(1, 41)]
USER_CATEGORICAL = ['gender', 'platform', 'age', 'location']
ITEM_CATEGORICAL = ['taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id']
