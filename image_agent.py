import base64
from openai import OpenAI
import google.generativeai as genai
from PIL import Image
import httpx
import os
import json
from tqdm import tqdm
import re
import socks
import socket
import requests
import subprocess
from google.oauth2 import service_account
import google.auth.transport.requests
import string

questions_visual = [
    # --- 1. 肿瘤位置 ---
    # 临床意义: 评估肿瘤负荷和手术可切除性。
    ['Tumor is confined to a single hepatic segment.',
     'Tumor involves multiple hepatic segments or is multifocal.'],

    # --- 2. 腹水 ---
    # 临床意义: 常提示肝硬化、门静脉高压或腹膜转移，是预后不良的标志。
    ['No evidence of ascites (free fluid) in the abdominal cavity.',
     'Ascites is present in the abdominal or pelvic cavity.'],

    # --- 3 问题1: 大血管癌栓 (准确率100%) --- 红
    # 表现良好，保持不变。
    ['No evidence of tumor thrombus in major portal or hepatic veins.', 
     'Clear evidence of enhancing soft tissue thrombus within a major portal or hepatic vein.'],

    # --- 4. 邻近脏器侵犯 ---
    # 临床意义: 判断肿瘤是否为局部晚期，影响分期和治疗决策。
    ['A clear fat plane exists between the tumor and adjacent organs (e.g., kidney, colon, diaphragm).',
     'The tumor directly invades or shows an indistinct border with an adjacent organ.'],

    # --- 5 问题2: 肝包膜回缩 (准确率84%) --- 红
    # 本次答错。微调措辞，使其更关注一个可观察的动作：“向内牵拉”。
    ['The contour of the adjacent liver capsule is smooth, without any inward pulling.', 
     'There is a focal, inward pulling or retraction of the adjacent liver capsule.'],

    # --- 6 问题3: 肿瘤包膜 (准确率52%) --- 红
    # 本次答错。放宽“完整且均匀”的严格定义，允许不完整的包膜。
    ['In the Portal Venous Phase (PVP), a distinct, bright enhancing rim along the lesion\'s border is absent.',
     'In the Portal Venous Phase (PVP), a distinct, bright enhancing rim (capsule) is visible along a significant portion of the lesion\'s border.'],

    # --- 7 问题4: 肿瘤边缘 (准确率52%) --- 红
    # 本次答错。简化提问方式，从判断“浸润性”转为判断“是否容易勾勒边界”。
    ['The lesion\'s border is predominantly sharp and clear, allowing for easy delineation.',
     'The lesion\'s border is predominantly blurry or irregular, making it difficult to delineate precisely.'],

    # --- 8. 脂肪/脂质 ---
    # 修改：移除对平扫的依赖，改为基于对比增强后信号的间接推断。
    ['The lesion shows signal characteristics consistent with typical soft tissue on AP and PVP scans.',
     'The lesion contains areas with signal characteristics highly suggestive of macroscopic fat (e.g., extreme low density that does not enhance).'],

    # --- 9. 出血 ---
    # 修改：移除对平扫的依赖，改为寻找对比增强图像上不寻常的高密度区域。
    ['The lesion does not show irregular, non-enhancing areas of high density suggestive of hemorrhage.',
     'The lesion contains irregular, non-enhancing hyperdense areas on AP or PVP scans, suggestive of recent hemorrhage.'],

    # --- 10. 主要成分 (囊性 vs. 实性) ---
    # 临床意义: 区分肿瘤是液体为主还是组织为主。HCC绝大多数是实性，而囊腺瘤/癌或脓肿则以囊性为主。
    ['The lesion is predominantly cystic (mostly fluid-filled, does not significantly enhance).',
     'The lesion is predominantly solid (mostly soft tissue, shows enhancement).'],

    # --- 11 问题5: 马赛克结构 (准确率32%) - 重点修改 --- 红
    # 修改原因: 将“马赛克”这个抽象概念，分解为“内部是否存在多个独立强化小结节”这个具体的结构性问题。
    ['In the Arterial Phase (AP), the lesion\'s internal enhancement is mostly uniform, without distinct smaller nodules inside.',
     'In the Arterial Phase (AP), the lesion contains multiple, distinct, smaller enhancing nodules within the larger mass ("nodule-in-nodule" appearance).'],

    # --- 12. 瘤周异常灌注 ---
    # 临床意义: 表现为动脉期肿瘤周围一过性的地图状、扇形或环形强化。这是由于肿瘤窃血或影响了局部血流动力学，是HCC的辅助特征。
    ['In the Arterial Phase (AP), no abnormal perfusion is seen in the liver parenchyma surrounding the lesion.',
     'In the Arterial Phase (AP), there is transient, wedge-shaped or halo-like hyperenhancement in the parenchyma around the lesion.'],

    # --- 13. 脂肪肝 ---
    # 临床意义: 肝细胞癌常发生在肝硬化或脂肪肝背景上。CT上表现为肝脏密度普遍低于脾脏。
    ['The background liver parenchyma does not show evidence of steatosis (liver density is greater than or equal to spleen density).',
     'The background liver parenchyma shows diffuse steatosis (liver density is distinctly lower than spleen density).'],

    # --- 14 问题6: 关于肿瘤内部坏死 --- 红
    # 临床意义: 大范围的缺血或坏死区域在CT上表现为无强化的低密度区。这通常见于生长迅速、血供不足的较大肿瘤。
    # 这个特征本身没有特异性，但在评估肿瘤的生物学行为时有参考价值。
    ['Lesion is predominantly enhancing with no significant non-enhancing areas.', 
     'Significant non-enhancing central areas, suggestive of intratumoral necrosis or ischemia, are present.'],

    # --- 15 问题7: 关于晕环征 (Halo Sign) --- 红
    # 准确率: 80%。表现良好。修改原因: 为确保严谨，明确指出这是PVP期的特征。
    ['In the Portal Venous Phase (PVP), no peritumoral hypodense halo (a dark ring) is observed.',
     'In the Portal Venous Phase (PVP), a distinct peritumoral hypodense halo (a dark ring around the tumor) is visible.'],
     
    # --- 16. 冠状强化 (Corona Enhancement) ---
    # 临床意义: 指在动脉晚期或门静脉期，肿瘤周边出现一圈强化的“冠冕”状血管影，并向心性流入。这是高分化HCC的一个特征。
    ['"Corona enhancement" is absent in the late arterial or portal venous phase.',
     '"Corona enhancement" is present, showing as a radiating vascular corona at the tumor periphery.'],

    # --- 17 问题8: 关于肿瘤内部动脉 --- 红
    # 准确率: 70%。表现尚可。修改原因: 明确这是AP期的特征，强化视觉线索。
    ['In the Arterial Phase (AP), distinct hyper-enhancing dot-like or linear structures are absent within the lesion.',
     'In the Arterial Phase (AP), distinct hyper-enhancing dot-like or linear structures (intratumoral arteries) are clearly visible.'],

    # --- 18 问题9(原第十题): 关于环形动脉期高强化 (Rim APHE) --- 红
    # 准确率: 80%。表现良好。修改原因: 为保持一致性，明确这是AP期的特征。
    ['In the Arterial Phase (AP), hyperenhancement is non-rim-like (e.g., it is diffuse or heterogeneous throughout the lesion).',
     'In the Arterial Phase (AP), the lesion demonstrates clear and unequivocal "Rim APHE" (enhancement is confined to the periphery).'],

    # --- 问题19: 强化模式的分类 (Washout vs. Fade) ---
    # 修改思路: 将问题从判断“有没有Fade”，变为一个二选一的分类题：“这个病灶的动态模式更像Washout还是Fade？”
    # 这让两个选项都变得很具体。
    ['The lesion demonstrates a "washout" pattern, becoming significantly less enhanced (hypodense) in the PVP compared to the AP.',
     'The lesion demonstrates a "fade" or "fill-in" pattern, remaining iso-dense or becoming more enhanced in the PVP compared to the AP.'],

    # --- 问题20: 内部结构的分类 (马赛克 vs. 结中结) ---
    # 修改思路: 将“有没有结中结”变为一个更明确的分类：“内部结构是无序的马赛克，还是有序的结中结？”
    # 这样，当模型看到一个杂乱的肿瘤时，可以自信地选择0。
    ['The AP enhancement is chaotically heterogeneous or mosaic-like, with multiple, similarly-enhancing nodules scattered without a clear hierarchy.',
     'The lesion shows a distinct "nodule-in-nodule" architecture, with a smaller hyper-enhancing nodule clearly nested within a larger, less-enhancing parent lesion.'],

    # --- 问题21: 中心区域的动态变化 (坏死/廓清 vs. 渐进性强化) ---
    # 修改思路: 将“有没有中心强化”变为一个更明确的对立性问题：“肿瘤中心是在PVP期变暗了，还是变亮了？”
    ['The central part of the lesion shows "central washout" or remains as a non-enhancing necrotic area from AP to PVP.',
     'The central part of the lesion shows "progressive enhancement", becoming distinctly brighter in the PVP than it was in the AP.']
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
You are to act as an expert radiological pattern analyst. Your objective is to perform a detailed, systematic analysis of visual patterns within a provided sequence of CT images for a quantitative academic research project. 
**CRITICAL NOTE: This task is for non-clinical, computational pattern recognition ONLY. It is NOT for medical diagnosis, advice, or any clinical application.**

### Image Sequence Context
You will be analyzing a sequence of **20 cross-sectional CT images**, structured as two distinct phases. It is essential you understand this structure:

- **Image Sequence Structure:**
  - **Images 1-10: Arterial Phase (AP) Sequence.** This phase is critical for evaluating the vascularity of the lesion (e.g., hyperenhancement, internal arteries).
  - **Images 11-20: Portal Venous Phase (PVP) Sequence.** This phase is crucial for assessing features like the tumor capsule, surrounding liver enhancement, and dynamic patterns like "washout".

- **Image Pre-processing:** The images have been pre-processed with a specific mask. The **only visible object** is a single liver lesion (the tumor) and its immediate peritumoral margin. All other anatomical structures have been blacked out.

- **Target of Analysis:** Your entire analysis must focus exclusively on this **single, visible lesion** as it appears across both the AP and PVP sequences.

### Analytical Framework
For each question, your reasoning MUST explicitly apply this simplified framework. **Your analysis must be supported by citing the specific image filenames that best demonstrate the feature in question.**

**1. Overall Shape & Phases:** Perceive the lesion as two 3D objects (one AP, one PVP). Your initial goal is to understand the lesion's overall shape in each phase.
**2. Boundary & Capsule:** Analyze the lesion's edge. Is it sharp or irregular? Look for a bright, enhancing rim (capsule), paying special attention to the PVP sequence.
**3. Internal Texture:** Examine the lesion's internal pattern, focusing on the AP. Is it uniform (homogeneous) or patchy (heterogeneous)? Specifically identify:
    - Large **dark areas** (suggesting necrosis).
    - Bright **dots or lines** (suggesting intratumoral arteries, especially in AP).
    - A **"nodule-in-nodule"** or mosaic appearance (especially in AP).
**4. Verification:** Ensure any feature you identify is consistent across several adjacent slices. Base your final conclusion on clear, verifiable patterns observed in the images.

### Instructions
1.  Perform a holistic analysis of the entire 20-image sequence according to the framework above.
2.  For each of the {len(questions_list)} visual patterns listed below, you must provide a structured answer containing three parts:
    - **"answer"**: Your final choice, either **0** for the first description or **1** for the second.
    - **"confidence"**: Your confidence in this answer, as an integer from **1 (very uncertain)** to **5 (very certain)**.
    - **"justification"**: A brief, single-sentence explanation of your reasoning. **This explanation is mandatory and must explicitly reference the key visual feature and the primary image numbers (e.g., "AP images 4-7", "PVP images 15-18") or sequences (e.g., "comparing AP to PVP") that support your decision.**
3.  Your entire response MUST be a single, raw JSON object. The top-level keys should be `pattern_1`, `pattern_2`, etc.

### Absolute Output Formatting Rules
You must follow these two rules precisely:
1.  **RAW JSON ONLY:** Your entire response MUST start with the opening brace `{{` and end with the closing brace `}}`. Do NOT output any text, summaries, or markdown like ```` ```json ```` before or after the JSON object.
2.  **CLEAN CONTENT:** The JSON's content must be clean. Use only standard printable characters and avoid all invisible control characters or non-standard whitespace (like U+00A0).

### Visual Patterns to Evaluate:
{formatted_questions}
### Required JSON Output Format (Example):
```json
{{
    "pattern_3": {{
      "answer": 0,
      "confidence": 5,
      "justification": "For this negative finding, the major vessels were reviewed across all relevant slices and no enhancing tissue or filling defects were identified within them."
    }},
    "pattern_5": {{
      "answer": 1,
      "confidence": 3,
      "justification": "For this uncertain finding, a shallow, subtle indentation of the capsule is suggested in PVP images 14-15, but it is not definitive enough to be classified as a clear retraction."
    }},   
    "pattern_10": {{
      "answer": 1,
      "confidence": 5,
      "justification": "The lesion's internal tissue demonstrates significant contrast uptake and enhancement across the AP and PVP sequences, confirming its predominantly solid composition."
    }},
    "pattern_15": {{
      "answer": 0,
      "confidence": 5,
      "justification": "For this negative finding, review of the PVP sequence (images 11-20) reveals a sharp enhancing capsule, not a hypodense ring characteristic of a halo sign."
    }}
}}
"""
    return prompt_text


# 将一张图片文件转换成一串文本字符串
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gcp_access_token():
    """通过gcloud命令行工具获取一个临时的访问令牌。"""
    try:
        # 确保gcloud已配置好正确的项目
        token = subprocess.check_output(['gcloud', 'auth', 'print-access-token']).decode('utf-8').strip()
        return token
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误:无法获取GCP访问令牌。请确保您已安装gcloud CLI并运行了 'gcloud auth application-default login'。")
        return None


def get_scores_from_llm(image_paths: list, prompt_text: str, model_name: str = 'gpt-4o'):
    """
    一次性将【多张】图片和文本发送给指定的LLM (GPT或Gemini) 以获取评分。

    Args:
        image_paths (list): 包含多张图片文件路径的列表。
        prompt_text (str): 提示词文本。
        model_name (str): 要使用的模型名称，例如 'gpt-4o', 'gemini-2.5-pro'。

    Returns:
        dict: 解析后的JSON分数,如果失败则返回None。
    """
    content = None # 初始化 content 变量
    client = None

    # --- 代理设置 (只对OpenAI/兼容API生效) ---
    proxy_url = "socks5://hkumedai:bestpaper@66.42.72.8:54321"
    http_client = httpx.Client(proxies=proxy_url)

    # --- 根据模型名称选择不同的API调用逻辑 ---
    if 'gpt' in model_name.lower():
        # --- OpenAI API 的逻辑 ---
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("错误：请先设置您的 OPENAI_API_KEY 环境变量。")
            return None
        
        # OpenAI客户端会自动使用代理
        client = OpenAI(api_key=api_key, http_client=http_client)
        
        # 构建包含多张图片的消息体
        message_content = [{"type": "text", "text": prompt_text}]
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
        
        try:
            print(f"正在将 {len(image_paths)} 张图片一次性发送给 {model_name}...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": message_content}],
                max_tokens=8192,
                temperature=0.0
            )
            content = response.choices[0].message.content

        except Exception as e:
            print(f"调用 OpenAI API 时发生错误: {e}")
            return None

    elif 'gemini' in model_name.lower():
        """
        使用【服务账号JSON密钥】调用Vertex AI上的Gemini模型。
        """
        # --- Vertex AI 的配置 ---
        project_id = "gen-lang-client-0514315738"
        location = "us-central1"
        
        # --- (这是关键修改：使用服务账号进行身份验证) ---
        try:
            # 1. 从JSON文件加载凭据
            credentials = service_account.Credentials.from_service_account_file(
                '/home/yxcui/FM-Bridge/testing_file/gen-lang-client-0514315738-faeaf04b384e.json', 
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # 2. 使用凭据获取一个临时的访问令牌
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            access_token = credentials.token
            
        except Exception as e:
            print(f"从服务账号文件进行身份验证时出错: {e}")
            return None

        # 构建Vertex AI的REST API端点URL
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:generateContent"

        # 构建请求头，使用Bearer Token进行授权
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        # 构建请求体 - 修正格式
        parts = [{"text": prompt_text}]
        for image_path in image_paths:
            parts.append({
                "inline_data": { 
                    "mime_type": "image/png", 
                    "data": encode_image_to_base64(image_path) 
                }
            })
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": { 
                "temperature": 0.0, 
                "maxOutputTokens": 8192 
            }
        }
        
        try:
            print(f"正在将 {len(image_paths)} 张图片一次性发送给 Vertex AI 上的 {model_name}...")
            proxies = { 'https': 'socks5://hkumedai:bestpaper@66.42.72.8:54321' }
            response = requests.post(url, headers=headers, json=payload, proxies=proxies)
            
            # 打印详细的错误信息
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            response.raise_for_status()
            
            # 解析响应 - 修正格式
            response_data = response.json()
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                content = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                print("Error: No candidates in response")
                print(f"Response: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"调用 Vertex AI API 时发生网络错误: {e}")
            return None
        except Exception as e:
            print(f"处理 Vertex AI 响应时发生未知错误: {e}")
            return None


    # 如果还有解析问题，用下面的来检查问题
    # --- 通用的、健壮的JSON解析逻辑 (V-Final) ---
    # 增加边界校验和字符串清洗步骤
    """if not content:
        print("错误: API没有返回任何内容。")
        return None

    cleaned_string = ""
    try:
        # 步骤 1: 优先尝试从Markdown代码块中提取
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            # 步骤 2: 如果没有Markdown，则手动查找从第一个'{'到最后一个'}'的边界
            start_index = content.find('{')
            end_index = content.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                # 只截取到最后一个 '}'，有效“切掉”之后可能存在的任何“幽灵字符”
                json_string = content[start_index : end_index + 1]
            else:
                print(f"错误: 在回复中未找到有效的JSON对象边界。原始回复内容: {content}")
                return None

        # 步骤 3: 清洗字符串 (核心步骤)
        cleaned_string = json_string.replace('\u00A0', ' ')

        # 步骤 4: 使用清洗后的字符串进行解析
        return json.loads(cleaned_string)

    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON失败。错误信息: {e}")
        print("--- 导致失败的、被清洗过的JSON字符串 ---")
        print(cleaned_string) # 打印出清洗后的字符串，用于最终调试
        print("------------------------------------")
        return None"""

    if not content:
        print("错误: API没有返回任何内容。")
        return None
    """
    一个最终版的、带有“强力消毒”功能的健壮函数。
    它会过滤掉所有非标准的、不可见的控制字符。
    """
    cleaned_string = ""
    try:
        # 步骤 1: 提取JSON字符串
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            start_index = content.find('{')
            end_index = content.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string = content[start_index : end_index + 1]
            else:
                print(f"错误: 在回复中未找到有效的JSON对象边界。原始回复内容: {content}")
                return None

        # 步骤 2: 清洗常规的非法空格
        cleaned_string = json_string.replace('\u00A0', ' ')

        # 步骤 3: 强力消毒 - 只保留“可打印”的ASCII字符
        # 这是解决本次问题的核心步骤
        sanitized_string = ''.join(char for char in cleaned_string if char in string.printable)

        # 步骤 4: 使用经过“消毒”的字符串进行解析
        return json.loads(sanitized_string)

    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON失败。错误信息: {e}")
        # 在失败时打印出经过两轮清洗和消毒的字符串，用于最终调试
        print("--- 导致失败的、经过最终消毒的JSON字符串 ---")
        # 确保sanitized_string在打印前已被定义
        if 'sanitized_string' in locals():
            print(sanitized_string)
        else:
            print(cleaned_string)
        print("-----------------------------------------")
        return None



# ---------------------------------------------------
# main
base_cropped_path = "/home/yxcui/FM-Bridge/testing_file/test_dataset/cropped_20_slices_image"
all_results = {}

try:
    patient_ids = [d for d in os.listdir(base_cropped_path) if os.path.isdir(os.path.join(base_cropped_path, d))]
    patient_ids.sort() # 确保处理顺序
    if not patient_ids:
        print(f"错误：在文件夹 '{base_cropped_path}' 中没有找到任何病人文件夹。")
        exit()
    print(f"找到了 {len(patient_ids)} 个病人的已裁剪图片文件夹，准备开始处理...")
except FileNotFoundError:
    print(f"错误：找不到文件夹 '{base_cropped_path}'。请检查路径是否正确。")
    exit()

for patient_id in tqdm(patient_ids, desc="Processing Patients"):
    
    # 构建当前病人存放20张裁剪后图片的文件夹路径
    patient_folder = os.path.join(base_cropped_path, patient_id)
    
    # 收集该文件夹下所有20张图片的完整路径
    try:
        twenty_image_paths = sorted([os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('.png')])
        if len(twenty_image_paths) == 0:
            print(f"警告：病人 {patient_id} 的文件夹是空的，跳过。")
            continue
    except FileNotFoundError:
        print(f"警告：找不到病人 {patient_id} 的文件夹，跳过。")
        continue

    # 调用统一的函数，将20张图片的路径列表一次性传入
    # 在这里选择您想使用的模型: 'gpt-4o' 或 'gemini-2.5-pro'
    prompt = create_visual_pattern_prompt(questions_visual)
    result = get_scores_from_llm(
        image_paths=twenty_image_paths, 
        prompt_text=prompt,
        model_name='gemini-2.5-pro' 
    )

    # 收集结果
    all_results[patient_id] = result

# 打印并保存最终结果
print("\n\n--- 所有处理已完成 ---")
print("最终结果汇总:")
for patient_id, result in all_results.items():
    print(f"  - 病人ID {patient_id}: {result}")

# 只包含那些 result 不是 None 的条目
filtered_results = {
    patient_id: result 
    for patient_id, result in all_results.items() 
    if result is not None
}

with open('agent_results_20_average.json', 'w') as f:
    json.dump(filtered_results, f, indent=4)

