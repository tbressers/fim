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

def ask_llm(prompt):
    if not isinstance(prompt, list):
      prompt = tokenizer(prompt).input_ids

    model.eval()
    outputs = model.generate(
        input_ids=torch.tensor([prompt]).cuda(),
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True,
    )
    return (tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])

def get_completion(prefix, suffix, prompt = None):
    if prompt == None:
      prompt = f"""<|im_start|>user<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"""
    return ask_llm(prompt)

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


print("\n\n")

prefix = """
    SplineFitter(int segmentCount,
            int polynomialOrder,
            double beta) {
        this.segmentCount = segmentCount;
        this.polynomialOrder = polynomialOrder;
        this.beta = beta;
    }

    /**
     * Creates a new SplineFitter for existing segments/breaks.
     * 
     * @param breaks
     *            segments.
     * @param polynomialOrder
     *            order of the polynomial.
     * @param beta
     *            value for robust fitting (recommended: 0).
     */
    public SplineFitter(List<Double> breaks,
            int polynomialOrder,
            double beta) {
        this.breaks = breaks;
        this.segmentCount = breaks.size() - 1;
"""

suffix = """
        this.beta = beta;
    }
"""

output = get_completion(prefix, suffix)

print("\n\nOUTPUT:")
print(output)
print("""\n
Expected output:
        this.polynomialOrder = polynomialOrder;
\n
""")

#question = "What is the recommended beta for the SplineFitter function?"
#answer = ask_llm(question)
#rint(answer)
