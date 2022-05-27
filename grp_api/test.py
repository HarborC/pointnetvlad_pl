from tqdm import tqdm
import numpy as np
from gpr.dataloader import PittsLoader
from gpr.evaluation import get_recall
from feature import PointNetVladFeature
from option import get_options

hparams = get_options()

# * Test Data Loader, change to your datafolder
pitts_loader = PittsLoader("/media/s1/cjg/GRP/GPR_Competition/datasets/TEST")
# pitts_loader = PittsLoader('../datasets/gpr_pitts_sample/')

# * Point cloud conversion and feature extractor
PNV_fea = PointNetVladFeature("/media/s1/cjg/logs/exp/version_None/checkpoints/epoch=54-step=84865.ckpt", hparams.prefixes_to_ignore)


# feature extraction
feature_ref = []
for idx in tqdm(range(len(pitts_loader)), desc = 'comp. fea.'):
    pcd_ref = []
    
    pcd_ = pitts_loader[idx]['pcd']
    pcd_ref.append(pcd_)

    # You can use your own method to extract feature
    feature_ref.append(PNV_fea.infer_data(pcd_ref))

# print(feature_ref)

# evaluate recall
feature_ref = np.array(feature_ref)
topN_recall, one_percent_recall = get_recall(feature_ref, feature_ref)

print("topN_recall", topN_recall)
print("one_percent_recall", one_percent_recall)

from gpr.tools import save_feature_for_submission
save_feature_for_submission('pnv.npy', feature_ref)