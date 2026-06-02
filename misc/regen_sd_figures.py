"""Regenerate SD curve figures from existing JSON without re-running simulations."""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")

def load_json(p):
    with open(p) as f:
        return json.load(f)

def fix_keys(sd):
    """Convert JSON string keys back to float."""
    return {
        float(d): {
            pk: {float(pw): v for pw, v in pws.items()}
            for pk, pws in pks.items()
        }
        for d, pks in sd.items()
    }

# MRG
from experiments_v2.mrg_validation import fig_sd_curves as mrg_sd
sd = fix_keys(load_json(ROOT / "outputs/mrg_validation/data_mrg_sd.json"))
mrg_sd(sd)
print("MRG SD curves done")

# Rattay
from experiments_v2.rattay_validation import fig_sd_curves as rattay_sd
sd = fix_keys(load_json(ROOT / "outputs/rattay_validation/data_rattay_sd.json"))
rattay_sd(sd)
print("Rattay SD curves done")

# Sundt
from experiments_v2.sundt_validation import fig_sd_curves as sundt_sd
sd = fix_keys(load_json(ROOT / "outputs/sundt_validation/data_sundt_sd.json"))
sundt_sd(sd)
print("Sundt SD curves done")

# Sweeney
from experiments_v2.sweeney_validation import fig_sd_curves as sweeney_sd
sd = fix_keys(load_json(ROOT / "outputs/sweeney_validation/data_sweeney_sd.json"))
sweeney_sd(sd)
print("Sweeney SD curves done")

print("All done.")
