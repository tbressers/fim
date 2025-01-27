import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=None,
    device_map=None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.cuda()

# Replace with your adapter ID
#adapter_id = "./output/Qwen/Qwen2.5-1.5B-Instruct/checkpoint-1000"
adapter_id = "./last_run/checkpoint-1000"
# Latest revision
revision = None
model = PeftModel.from_pretrained(model, adapter_id, revision=revision, adapter_name="my-adapter")
model.set_adapter("my-adapter")

stop_token = "<|im_end|>" # eos_token
#stop_token = "<|eot_id|>"
#stop_token = "\\n"
stop_token_id = tokenizer.encode(stop_token)[0]

def get_completion(prefix, suffix, prompt = None):
    if prompt == None:
      prompt = f"""<|im_start|>user<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"""
    if not isinstance(prompt, list):
      prompt = tokenizer(prompt).input_ids

    model.eval()
    outputs = model.generate(
        input_ids=torch.tensor([prompt]).cuda(),
        max_new_tokens=256,
        temperature=0.4,
        do_sample=True,
        eos_token_id=stop_token_id,
        pad_token_id=4
    )
    return (tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])

prefix = """
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.context.DataContainerAwareFlowContext;
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.domain.AbfDomainModelUtils;
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.flowelement.CalculateAstigmatismPerPoint;
"""

suffix = """
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.taskadapter.CopyDataSet;
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.taskadapter.FillLevelingMa;
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.taskadapter.FilterAstigmatismEdgeFields;
"""

output = get_completion(prefix, suffix)

print("\n\nOUTPUT:")
print(output)
print("""\n
Expected output:
import com.asml.wfa.hvmintegration.functions.abf.businesslogic.calculationflow.flowelement.CalculateAstigmatismPerSlit;
\n
""")