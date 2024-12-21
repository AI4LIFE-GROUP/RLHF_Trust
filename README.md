# Source Code For the ICLR Submission "More RLHF, More Trust? On The Impact of Preference Alignment On Trustworthiness"

## Project Structure

```
.
├── README.md
├── attribution/
    ├── convert_lora.py
    ├── attribute_sft.py
    ├── attribute_ppo.py
    ├── attribute_dpo.py
├── data/
    ├── toxicity/
    ├── stereotype/
    ├── machine_ethics/
    ├── truthfulness/
    ├── privacy/
├── toxicity_eval/
├── bias_eval/
├── ethics_eval/
├── truthfulness_eval/
├── privacy_eval/
```

## Description of the Repo

For RLHF, we rely on existing implementations from https://github.com/eric-mitchell/direct-preference-optimization and https://github.com/lauraaisling/trlx-pythia/.

The evaluation code for each trustworthiness aspect is included in the corresponding `_eval/` directory.

To reproduce the results, first run the files end with `_exp.py` to run the evaluation and store the results, and then run the files end with `_analysis.py` to parse the model generations and compute the final scores. Remember to replace `...` with actual model paths.

The code for conducting our data attribution is included in the `attribution/` directory, which consists of the attribution code for SFT, PPO, and DPO, as well as the code to convert our fully fine-tuned models to Lora based models.

