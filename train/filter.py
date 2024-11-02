import argparse
from collections import defaultdict
import json
import numpy as np
import torch

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument(
    "--load_path",
    help="path to load data",
    default="/root/aspect_sum/scr/annotation/reviwes/label_val.pt",
)
parser.add_argument(
    "--save_path",
    help="path to save filtered data",
    default="/root/aspect_sum/scr/annotation/reviwes/label_val_kept.pt",
)

parser.add_argument(
    "--save_path_json",
    help="path to save filtered data",
    default="/root/aspect_sum/scr/annotation/reviwes/label_val_kept.json",
)
args = parser.parse_args()

res_pt = torch.load(args.load_path) 

## filtering
variation_rate_list = res_pt["variation_rate"]
print(len(variation_rate_list))
threshold = np.percentile(variation_rate_list, 90)
#90번째 백분위수 계산 = 상위 10%해당되는 값 
print(threshold)
kept, filtered = defaultdict(list), defaultdict(list)

for labels, origin, comp, retrieval, cr,vr,hr,mr, ag,ap in zip(
    res_pt["labels"],
    res_pt["origin"],
    res_pt["comp"],
    res_pt["retrieval"],
    res_pt["comp_rate"],
    res_pt["variation_rate"],
    res_pt["hitting_rate"],
    res_pt["matching_rate"],
    res_pt["alignment_gap"],
    res_pt['aspects'],
):
    if vr >= threshold:
        filtered["labels"].append(labels)
        filtered["origin"].append(origin)
        filtered["comp"].append(comp)
        filtered["retrieval"].append(retrieval) #실제 검색된 토큰 
        filtered["comp_rate"].append(cr) #압축률
        filtered["variation_rate"].append(vr) #압축된 테스트에서 검색되지않은 비율 -> 낮을수록 좋음 
        filtered["hitting_rate"].append(hr) #원본 텍스트에서 검색된 토큰의 비율
        filtered["matching_rate"].append(mr) #레이블이 1인 경우의 비율
        filtered["alignment_gap"].append(ag) #원본 토큰 비율과 레이블 비율 차이 = 신뢰성 평가 
        filtered['aspects'].append(ap)
    else:
        kept["labels"].append(labels)
        kept["origin"].append(origin)
        kept["comp"].append(comp)
        kept["retrieval"].append(retrieval)
        kept["comp_rate"].append(cr)
        kept["variation_rate"].append(vr)
        kept["hitting_rate"].append(hr)
        kept["matching_rate"].append(mr)
        kept["alignment_gap"].append(ag)
        kept['aspects'].append(ap)

alignment_gap_list = kept["alignment_gap"]
threshold = np.percentile(alignment_gap_list, 90)
kept2 = defaultdict(list)

filtered_count = 0
kept2_count = 0

for labels, origin, comp, retrieval, cr, vr, hr, mr, ag,ap in zip(
    kept["labels"],
    kept["origin"],
    kept["comp"],
    res_pt["retrieval"],
    kept["comp_rate"],
    kept["variation_rate"],
    kept["hitting_rate"],
    kept["matching_rate"],
    kept["alignment_gap"],
    kept['aspects'],
):
    if ag >= threshold:
        filtered["labels"].append(labels)
        filtered["origin"].append(origin)
        filtered["comp"].append(comp)
        filtered["retrieval"].append(retrieval)
        filtered["comp_rate"].append(cr)
        filtered["variation_rate"].append(vr)
        filtered["hitting_rate"].append(hr)
        filtered["matching_rate"].append(mr)
        filtered["alignment_gap"].append(ag)
        filtered['aspects'].append(ap)
        filtered_count += 1 
    else:
        kept2["labels"].append(labels)
        kept2["origin"].append(origin)
        kept2["comp"].append(comp)
        kept2["retrieval"].append(retrieval)
        kept2["comp_rate"].append(cr)
        kept2["variation_rate"].append(vr)
        kept2["hitting_rate"].append(hr)
        kept2["matching_rate"].append(mr)
        kept2["alignment_gap"].append(ag)
        kept2['aspects'].append(ap)
        kept2_count += 1 
        
print(f"Filtered count: {filtered_count}")
print(f"Kept count: {kept2_count}")

torch.save(kept2, args.save_path)
json.dump(kept2, open(args.save_path_json, "w"), indent=4)

        
        
    
