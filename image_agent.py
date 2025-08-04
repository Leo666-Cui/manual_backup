import base64
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import httpx
import os
import json
from tqdm import tqdm
import socks
import socket

# 中文注释：这是转化后的视觉模式问题列表
# 每个问题都聚焦于几何、纹理或密度这些客观特征
questions_visual = [
    # 1. 大血管癌栓 -> 视觉上表现为血管内有无低密度充盈缺损
    ['In major blood vessels, the contrast filling is uniform, with no persistent low-density areas inside.',
     'In a major blood vessel, a distinct, persistent low-density area is visible, creating a filling defect within the contrast.'],

    # 2. 肝包膜回缩 -> 视觉上表现为肝脏外缘轮廓有无局部凹陷
    ['The outer contour of the liver maintains its expected smooth and convex (outwardly curved) shape.',
     'The outer contour of the liver shows a localized flattening or a distinct concave (inwardly curved) indentation.'],

    # 3. 包膜是否完整 -> 视觉上表现为病灶周围有无一条连续的环形线条
    ['A distinct, continuous rim or line separating the lesion from the liver tissue is not visible around the entire circumference.',
     'A distinct, continuous rim or line is visible, encapsulating the entire circumference of the lesion.'],

    # 4. 肿瘤边缘是否光滑 -> 视觉上表现为病灶边界是规则还是不规则
    ['The boundary of the lesion is ill-defined, irregular, or has spiculated (spiky) extensions.',
     'The boundary of the lesion is well-defined with a regular, smooth contour.'],

    # 5. 马赛克结构 -> 视觉上表现为病灶内部是均匀还是由不同密度区域组成
    ['The internal texture of the lesion is largely homogeneous (uniform in density).',
     'The internal texture of the lesion is heterogeneous, composed of multiple compartments or nodules with varying densities, creating a mosaic-like pattern.'],

    # 6. 严重缺血或坏死 -> 视觉上表现为病灶内有无不强化的低密度核心区
    ['Within the lesion, there are no significant, non-enhancing low-density areas after contrast administration.',
     'A significant, non-enhancing low-density area is present within the lesion, typically in the center.'],

    # 7. 低密度晕环 -> 视觉上表现为病灶紧邻外周有无一圈低密度环
    ['No distinct low-density halo or ring is observed immediately surrounding the lesion.',
     'A distinct low-density halo or ring is observed immediately surrounding the lesion.'],

    # 8. 内部动脉 -> 视觉上表现为动脉期病灶内有无点状或线状的强化结构
    ['During the arterial phase, no dot-like or linear enhancing structures are visible inside the lesion.',
     'During the arterial phase, dot-like or linear enhancing structures are visible inside the lesion.'],

    # 9. 门静脉癌栓(TTPVI) -> 视觉上表现为门脉内的栓子本身在动脉期是否强化
    ['A filling defect (thrombus) in the portal vein does not itself show significant enhancement in the arterial phase.',
     'A filling defect (thrombus) in the portal vein itself shows significant enhancement (becomes brighter) in the arterial phase.'],
    
    # 10. 边缘动脉期超强化(Rim APHE) -> 视觉上表现为病灶边缘在动脉期是否呈环形高亮
    ['The periphery of the lesion does not show a distinct, ring-like pattern of hyper-enhancement in the arterial phase.',
     'The periphery of the lesion shows a distinct, ring-like pattern of hyper-enhancement (appears significantly brighter) in the arterial phase.'],

    # 11. 周边廓清(Peripheral washout) -> (已适配单期相) 视觉上表现为延迟期病灶边缘是否比肝实质暗
    ['In the delayed phase image, the periphery of the lesion is not hypodense (darker) compared to the surrounding liver tissue.',
     'In the delayed phase image, the periphery of the lesion appears hypodense (darker) compared to the surrounding liver tissue.']
]

def create_visual_pattern_prompt(questions_list):
    """
    根据一个包含“视觉模式”问题配对的列表,为英文LLM生成一个结构化Prompt。
    """
    formatted_questions = ""
    for i, question_pair in enumerate(questions_list):
        # 提取特征名称的逻辑可以保持不变或简化，因为问题本身已经很具体
        # 为了简洁，我们直接用序号
        formatted_questions += f"**Pattern {i+1}:**\n"
        formatted_questions += f'   - (0) "{question_pair[0]}"\n'
        formatted_questions += f'   - (1) "{question_pair[1]}"\n\n'

    # --- 为视觉分析任务优化的新Prompt模板 ---
    prompt_text = f"""
You are an advanced image analysis assistant. Your task is to meticulously describe visual patterns within the provided image montage, based on a set of descriptions. This is for an academic research project on pattern recognition ONLY.

**Instructions:**
1.  **Identify the Liver and Target Lesion:** First, locate the liver in the provided CT images. Within the liver, identify the single most prominent lesion to be your target for analysis. If no lesion is visible, the healthy liver tissue is your target.
2.  For each of the {len(questions_list)} visual pattern pairs listed below, select the single description (0 or 1) that most accurately describes the patterns visible in **your defined target**.
3.  Your analysis should be based solely on geometric shapes, density (brightness/darkness), and textures.
4.  Provide your answers in a structured JSON format ONLY. The key for each item should be `pattern_n` (e.g., `pattern_1`, `pattern_2`).
5.  The value for each key must be an integer: **`0` for the first description** or **`1` for the second description**.
6.  Do not include any additional text, explanations, or conversational filler.

**Visual Patterns to Evaluate:**
{formatted_questions}
**Required JSON Output Format (Example):**
{{
  "pattern_1": 0,
  "pattern_2": 0,
  "pattern_3": 1,
  "pattern_4": 1,
  "pattern_5": 0,
  "pattern_6": 1,
  "pattern_7": 1,
  "pattern_8": 1,
  "pattern_9": 0,
  "pattern_10": 1,
  "pattern_11": 0
}}
"""
    return prompt_text



# 将一张图片文件转换成一串文本字符串
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_scores_from_llm(image_path, prompt_text, model_name='gpt-4o'):
    """
    根据指定的模型(GPT-4o或Gemini)从图像和文本中获取评分。
    此函数依赖于在主程序中设置的全局SOCKS代理。

    Args:
        image_path (str): 图像文件的路径。
        prompt_text (str): 提示词文本。
        model_name (str): 要使用的模型名称，例如 'gpt-4o', 'gemini-2.5-pro'。

    Returns:
        dict: 解析后的JSON分数,如果失败则返回None。
    """
    content = None # 初始化 content 变量


    if 'gpt' in model_name.lower():
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("错误：请先设置您的 OPENAI_API_KEY 环境变量。")
            return None
        
        base_url = "https://api.openai.com/v1"
        
        try:
            # 客户端初始化不再需要代理设置
            client = OpenAI(api_key=api_key, base_url=base_url)
            base64_image = encode_image_to_base64(image_path)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0
            )
            content = response.choices[0].message.content

        except Exception as e:
            print(f"调用 OpenAI API 时发生错误: {e}")
            return None

    elif 'gemini' in model_name.lower():
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("错误：请先设置您的 GEMINI_API_KEY 环境变量。")
            return None

        try:
            # 客户端初始化不再需要代理设置
            genai.configure(api_key=api_key)
            img = Image.open(image_path)
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content([prompt_text, img])
            content = response.text

        except Exception as e:
            print(f"调用 Gemini API 时发生错误: {e}")
            return None
    
    else:
        print(f"错误: 不支持的模型名称 '{model_name}'。请选择包含 'gpt' 或 'gemini' 的模型。")
        return None

    # --- 通用的JSON解析逻辑 ---
    if not content:
        print("错误: API没有返回任何内容。")
        return None
        
    try:
        # 尝试从回复中提取纯净的JSON字符串
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_string = content[start_index:end_index]
            scores = json.loads(json_string)
            return scores
        else:
            print(f"错误: 在LLM的回复中未找到有效的JSON对象。回复内容: {content}")
            return None
    except json.JSONDecodeError:
        print(f"错误: 解析JSON失败。原始回复内容: {content}")
        return None


# --- 完整调用流程 ---
# 代理服务器信息
SOCKS_PROXY_HOST = '66.42.72.8'
SOCKS_PROXY_PORT = 54321
SOCKS_PROXY_USER = 'hkumedai'
SOCKS_PROXY_PASS = 'bestpaper'

# 设置默认代理
socks.set_default_proxy(
    socks.SOCKS5, 
    SOCKS_PROXY_HOST, 
    SOCKS_PROXY_PORT, 
    username=SOCKS_PROXY_USER, 
    password=SOCKS_PROXY_PASS,
    rdns=True # rdns=True 等同于 socks5h，由代理服务器进行DNS解析
)

# 将 Python 的标准 socket 替换为代理 socket
socket.socket = socks.socksocket
print("--- SOCKS 全局代理已通过猴子补丁启用 ---")


base_montage_path = "/home/yxcui/FM-Bridge/testing_file/test_dataset/Montage"
all_results = {}

try:
    # os.listdir() 获取文件夹下所有文件名
    # 我们用一个列表推导式来筛选出所有以 .png 结尾的文件
    montage_files = [f for f in os.listdir(base_montage_path) if f.endswith('.png')]
    montage_files.sort()
    if not montage_files:
        print(f"错误：在文件夹 '{base_montage_path}' 中没有找到任何 .png 文件。")
        exit()
        
    print(f"找到了 {len(montage_files)} 张蒙太奇图片，准备开始处理...")

except FileNotFoundError:
    print(f"错误：找不到文件夹 '{base_montage_path}'。请检查路径是否正确。")
    exit()

for filename in tqdm(montage_files, desc="Processing Montage Images"):
    
    # 3.1 构建当前图片文件的完整路径
    montage_file_path = os.path.join(base_montage_path, filename)
    
    # 3.2 调用函数
    prompt = create_visual_pattern_prompt(questions_visual)
    # model_name='gemini-2.5-pro'
    result = get_scores_from_llm(montage_file_path, prompt)
    
    # 3.3 收集结果，我们用不带扩展名的文件名（例如病人ID）作为键
    patient_id = filename.replace('_montage.png', '')
    all_results[patient_id] = result

# 4. 打印最终收集到的所有结果
print("\n\n--- 所有处理已完成 ---")
print("最终结果汇总:")
for patient_id, result in all_results.items():
    print(f"  - 病人ID {patient_id}: {result}")

# 您可以在这里将 all_results 保存为 JSON 文件
with open('agent_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)

