from pathlib import Path
import json
from vllm_utils import VLLMServer
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn



server = VLLMServer(
    model_id = "allenai/OLMo-2-0425-1B",
    gpu = 0,
)

sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed":0,
}
cot_sampling_params = sampling_params.copy()
cot_sampling_params["stop"] = ["</answer>"]
cot_sampling_params["include_stop_str_in_output"] = True

data_path =  Path("data/gsm8k/test.jsonl")
output_path = Path("results/prompting_baselines.jsonl")
examples = []
results = []
categories_1 = [0,0,0]
categories_2 = [0,0,0]
categories_3 = [0,0,0]
with data_path.open("r", encoding = 'utf-8') as f:
    for line in f:
        example = json.loads(line)
        question = example['question']
        answer = example['answer']

        if '####' not in answer:
            raise ValueError("参考答案中缺少 #### 分隔符")
        ground_truth = answer.split('####', 1)[1].strip()
        examples.append({'question':question,'ground_truth':ground_truth})


prompt_dir = Path(__file__).resolve().parent / "prompts"
question_only = (prompt_dir / "question_only.prompt").read_text(encoding="utf-8")    
question_only = question_only.replace(r"\\boxed", r"\boxed")
zero_shot = (prompt_dir / "r1_zero.prompt").read_text(encoding="utf-8")
three_shot = (prompt_dir / "r1_zero_three_shot_gsm8k.prompt").read_text(encoding="utf-8")
templates = [question_only, zero_shot,three_shot]


try:
    server.start()
    for i in range(len(examples)):
        question = examples[i]["question"]
        ground_truth = examples[i]["ground_truth"]
        prompts = [mode.format(question = question) for mode in templates]
        completions = server.generate_completions(prompts = [prompts[0]], sampling_params = sampling_params)
        answer_1 = completions[0].text
        score_1 = question_only_reward_fn(response= answer_1, ground_truth= ground_truth)
        if score_1["reward"] == 1.0:
            categories_1[0] += 1
        elif score_1["format_reward"] == 1.0 and score_1["answer_reward"] == 0.0:
            categories_2[0] += 1
        elif score_1["format_reward"] == 0.0 and score_1["answer_reward"] == 0.0:
            categories_3[0] += 1
        completions = server.generate_completions(prompts = [prompts[1],prompts[2]], sampling_params = cot_sampling_params)
        answer_2 = completions[0].text
        answer_3 = completions[1].text
        score_2 = r1_zero_reward_fn(response= answer_2, ground_truth= ground_truth)
        score_3 = r1_zero_reward_fn(response= answer_3, ground_truth= ground_truth)
        if score_2["reward"] == 1.0:
            categories_1[1] += 1
        elif score_2["format_reward"] == 1.0 and score_2["answer_reward"] == 0.0:
            categories_2[1] += 1
        elif score_2["format_reward"] == 0.0 and score_2["answer_reward"] == 0.0:
            categories_3[1] += 1
        if score_3["reward"] == 1.0:
            categories_1[2] += 1
        elif score_3["format_reward"] == 1.0 and score_3["answer_reward"] == 0.0:
            categories_2[2] += 1
        elif score_3["format_reward"] == 0.0 and score_3["answer_reward"] == 0.0:
            categories_3[2] += 1
        results.append({
            "question_id": i+1,
            "question": question,
            "ground_truth": ground_truth,
            "prompts": prompts,
            "answer_1": answer_1,
            "answer_2": answer_2,
            "answer_3": answer_3,
            "score_1":score_1,
            "score_2":score_2,
            "score_3":score_3,
        })

finally:
    server.stop()
    print(categories_1)
    print(categories_2)
    print(categories_3)
    with output_path.open('a', encoding = 'utf-8') as f:
        for i in range(len(examples)):
            f.write(json.dumps(results[i], ensure_ascii=False)+'\n')
