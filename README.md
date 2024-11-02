# Abstractive Aspect-Based Comparative Summarization 

The proposed task in this paper aims to generate summaries not only of the
contrastive summaries but also comparative summaries of what the two entities have
in common by utilizing the entire review. The summarizer proposed in this paper is
based on a large language model.


## Quick Links
- **arXiv Paper**: ()
- **GitHub Repository**: ([https://github.com/your-repo-link](https://github.com/lluvecwonv/AACSum)



## run file
```bash
python main.py \
    --data_path "/root/AACS/coco/test_raw_reviews_content.json" \
    --model_family "FacebookAI/xlm-roberta-large" \
    --model "/root/AACS/models/xlm_roberta_large_reviews/checkpoint-700" \
    --result_path "/root/AACS/sumarization/coco_merge_sum" \
    --prompt "/root/AACS/templet/sum.txt" \
    --api_key "YOUR-API" \
    --seed 1212 \
    --assigner_model "gpt-4o-mini" \
    --assigner_prompt "/root/AACS/templet/gpt_assigner.txt" \
    --comp_text
```  

## contributor
Hyeon Jin, chaewon Yoon, yurim oh
