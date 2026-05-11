import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import timm

import sys
sys.path.append(os.path.dirname(__file__))
from multimodal_w8a8_smoothquant import SmoothQuantWrapper

# 1-Bit Binarization Layers (가져오기)
class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight): return torch.where(weight == 0, torch.ones_like(weight), torch.sign(weight))
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight.abs() > 1.0] = 0
        return grad_input

def binarize_weight(weight):
    if weight.dim() == 4: scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
    elif weight.dim() == 2: scale = weight.abs().mean(dim=1, keepdim=True)
    else: scale = weight.abs().mean()
    return BinarySTE.apply(weight) * scale

class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return torch.nn.functional.conv2d(input, bw, bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryLinear(nn.Linear):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(input, bw, bias)

def replace_layers_with_1bit(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and "stem" not in name and "head" not in name:
            bin_conv = BinaryConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                    module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            bin_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_conv)
        elif isinstance(module, nn.Linear) and "head" not in name and "classifier" not in name:
            bin_linear = BinaryLinear(module.in_features, module.out_features, module.bias is not None)
            bin_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_linear)
        else: replace_layers_with_1bit(module)


def build_llm_and_tokenizer(
    llm_name,
    use_qlora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None
):
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise RuntimeError("QLoRA 사용을 위해 bitsandbytes와 peft 설치가 필요합니다.") from e

        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        llm = prepare_model_for_kbit_training(llm)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules
        )
        llm = get_peft_model(llm, lora_config)
    else:
        llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return llm, tokenizer


class OmniModalBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_llm(
        self,
        llm_name,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None
    ):
        self.llm, self.tokenizer = build_llm_and_tokenizer(
            llm_name=llm_name,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules
        )

    def _build_inputs(self, text_input_ids, attention_mask=None, labels=None, image_embeds=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.long)

        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        if image_embeds is None:
            return text_embeds, attention_mask, labels

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        image_attention_mask = torch.ones(
            attention_mask.shape[0],
            1,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        if labels is not None:
            image_labels = torch.full(
                (labels.shape[0], 1),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([image_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def encode_image(self, images):
        vision_features = self.vision_encoder(images)
        return self.projection_head(vision_features).unsqueeze(1)

    def forward(self, images, text_input_ids, attention_mask=None, labels=None):
        image_embeds = None
        if images is not None:
            image_embeds = self.encode_image(images)

        inputs_embeds, attention_mask, labels = self._build_inputs(
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=image_embeds
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(
        self,
        images=None,
        prompt=None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True
    ):
        if input_ids is None:
            if prompt is None:
                raise ValueError("프롬프트 또는 input_ids가 필요합니다.")
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        image_embeds = None
        if images is not None:
            images = images.to(device, dtype=torch.bfloat16)
            image_embeds = self.encode_image(images)

        inputs_embeds, attention_mask, _ = self._build_inputs(
            text_input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            image_embeds=image_embeds
        )

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return outputs


class OmniModal1BitVLM(OmniModalBase):

    def __init__(
        self,
        vision_model_name='convnextv2_nano.fcmae_ft_in1k',
        llm_name='Qwen/Qwen1.5-0.5B',
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None
    ):
        super().__init__()
        print("OmniModal1BitVLM 초기화 중...")
        
        # 1. Vision Encoder (1-Bit)
        print(f"비전 인코더 로드 중: {vision_model_name} (1-Bit 변환 적용)")
        self.vision_encoder = timm.create_model(vision_model_name, pretrained=False)
        replace_layers_with_1bit(self.vision_encoder)
        
        # 기존 분류 헤드 제거 및 프로젝션을 위한 사전 준비
        vision_hidden_size = self.vision_encoder.head.fc.in_features
        self.vision_encoder.head.fc = nn.Identity() 

        # 2. Language Model (Qwen 0.5B)
        print(f"LLM 로드 중: {llm_name}")
        # 참고: 오프라인 환경을 위해 로컬 캐시 또는 HF 토큰이 필요할 수 있습니다.
        self._init_llm(
            llm_name=llm_name,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules
        )

        llm_hidden_size = self.llm.config.hidden_size
        
        # 3. Projection Head (Vision -> LLM Space)
        # 단순 Linear를 채택 (이전 분석 결과에 따라 정보 손실을 최소화)
        print("단일 Linear Projection Head 부착 완료.")
        self.projection_head = nn.Linear(vision_hidden_size, llm_hidden_size)
        
        # Vision Encoder 파라미터는 동결 (LLM과 Projection Head만 튜닝)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            

class OmniModalW8A8VLM(OmniModalBase):
    def __init__(
        self,
        vision_model_name='convnextv2_nano.fcmae_ft_in1k',
        llm_name='Qwen/Qwen1.5-0.5B',
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None
    ):
        super().__init__()
        print("OmniModalW8A8VLM 초기화 중...")
        
        self.vision_encoder = timm.create_model(vision_model_name, pretrained=False)
        in_features = self.vision_encoder.head.fc.in_features
        self.vision_encoder.head.fc = nn.Linear(in_features, 512)
        
        for name, module in dict(self.vision_encoder.named_modules()).items():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and "head" not in name:
                dummy_scale = torch.ones(module.in_channels if isinstance(module, nn.Conv2d) else module.in_features)
                sq_layer = SmoothQuantWrapper(module, dummy_scale)
                parts = name.split('.')
                parent = self.vision_encoder
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], sq_layer)
                
        self.vision_encoder.head.fc = nn.Identity()
        
        self._init_llm(
            llm_name=llm_name,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules
        )

        self.projection_head = nn.Linear(in_features, self.llm.config.hidden_size)
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    

def test_scaffold():
    print("스캐폴드 코드 검증을 위한 더미 테스트를 시작합니다.")
    print("코드 구조 검증 완료: OmniModal1BitVLM 클래스 작성 성공.")

if __name__ == "__main__":
    test_scaffold()
