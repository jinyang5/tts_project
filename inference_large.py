import torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # CPU 用 float32

repo = "parler-tts/parler-tts-large-v1"
revision = "refs/pr/9"  # 你想试 issue 里提到的 PR 分支就保留；不想用就删掉这个参数

model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo,
    revision=revision,          # 不想用 PR 就删掉这一行
    torch_dtype=dtype
).to(device).eval()

tok = AutoTokenizer.from_pretrained(repo)

desc   = "Enchanting the listener with her book reading, this woman with a Pakistani accent has a slightly higher voice and speaks quite fast. The voice is very close-sounding, and the recording is excellent."
prompt = "In a world where technology evolves faster than ever before, the power of communication has become the key to understanding, connection, and creativity. Every voice, every sound, and every story we share brings us one step closer to a more human future — one where ideas speak louder than boundaries."

ids_d = tok(desc, return_tensors="pt").input_ids.to(device)
ids_p = tok(prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    wav = model.generate(input_ids=ids_d, prompt_input_ids=ids_p)

sf.write("parler_tts_out.wav",
         wav.cpu().numpy().squeeze().astype("float32"),
         model.config.sampling_rate)
