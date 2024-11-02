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

## contributors
Thanks to these amazing contributors:

## Contributors

Thanks to these amazing contributors:

<a href="https://github.com/your-github-username">
  <img src="https://github.com/your-github-username.png" width="50" height="50" style="border-radius: 50%;" alt="Hyeon Jin">
</a> **Hyeon Jin**

<a href="https://github.com/lluvecwonv">
  <img src="https://github.com/lluvecwonv.png" width="50" height="50" style="border-radius: 50%;" alt="lluvecwonv">
</a> **chaewon yoon**

<a href="https://github.com/ohyurim1010">
  <img src="https://github.com/ohyurim1010.png" width="50" height="50" style="border-radius: 50%;" alt="yurim oh">
</a> **yurim oh**
