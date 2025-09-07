import os
import json
import base64
import requests
from tqdm import tqdm
from termcolor import cprint
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import re
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


# --- 1. 配置区 (Configuration Section) ---
# 请在此处修改您的个人配置

# 项目和模型配置
VERTEX_AI_CONFIG = {
    "project_id": "gen-lang-client-0514315738",
    "location": "us-central1",
    # 【【【重要】】】请务必修改为您的服务账号密钥文件的真实路径
    "credentials_path": "/home/yxcui/FM-Bridge/testing_file/gen-lang-client-0514315738-faeaf04b384e.json",
    "model_name": "gemini-2.5-pro"  # gemini-2.5-flash
}

# 代理配置
PROXY_CONFIG = {
    # 如果您不需要代理，请将此行设置为 None, 例如：'https': None
    'https': 'socks5://hkumedai:bestpaper@66.42.72.8:54321' 
}

# 【【【重要】】】请修改为您的数据集根目录
BASE_DATA_PATH = "/home/yxcui/FM-Bridge/testing_file/test_dataset/cropped_30_slices_image"


# 规则：定义每个特征最终诊断所必需的期相
FEATURE_PHASE_REQUIREMENTS = {
    "Enhancing Capsule": ["AP", "PVP", "DP"], # 44
    "Peritumoral Perfusion Alteration": ["AP", "PVP"], # 53
    "Peritumoral Hypodense Halo": ["PVP", "DP"], # 59
    "Corona Enhancement": ["AP", "PVP", "DP"], # 60
    "Custom TTPVI Feature": ["AP", "PVP", "DP"], # 62 
    "Fade Enhancement Pattern": ["AP", "PVP", "DP"], # 65
    "Nodule-in-Nodule Architecture": ["AP", "PVP", "DP"], # 66
    "Peripheral Washout": ["AP", "PVP", "DP"], # 68
    "Delayed Central Enhancement": ["AP", "DP"] # 70
}

# --- 临床问题定义 (Clinical Question Definitions) ---
# 将问题列表定义为全局常量，供所有Agent访问
FEATURE_DEFINITIONS = [
    {
        "name": "Enhancing Capsule", # 44
        "options": [
            'A distinct, hyper-enhancing rim is NOT identified in the PVP/DP, or any visible rim does not show clear enhancement compared to the AP.',
            'By comparing phases, a smooth rim is identified that enhances to become distinctly hyper-enhancing in the PVP or DP.'
        ]
    },
    {
        "name": "Peritumoral Perfusion Alteration", # 53
        "options": [
            'No clear perfusion anomalies are seen in the AP, or any observed hyperenhancement around the lesion persists into the PVP.',
            'A transient perfusion anomaly is confirmed: wedge-shaped or halo-like hyperenhancement is visible around the lesion in the AP and resolves (disappears) in the PVP.'
        ]
    },
    {
        "name": "Peritumoral Hypodense Halo", # 59
        "options": [
            'In the PVP or DP, the interface between the lesion and the surrounding liver parenchyma is direct, without a distinct, dark, intervening ring.',
            'In the PVP or DP, a distinct, dark (hypodense), well-defined ring is visible immediately surrounding the exterior of the lesion. This "halo" is darker than both the lesion\'s periphery and the adjacent liver tissue.'
        ]
    },
    {
        "name": "Corona Enhancement", # 60
        "options": [
            'No radiating vascular pattern is seen at the tumor periphery in any phase.',
            'A dynamic "corona enhancement" is identified: a radiating vascular pattern appears at the tumor periphery in the late AP or PVP and fades in later phases.'
        ]
    },
    {
        "name": "Custom TTPVI Feature", # 62
        "options": [
            'The combined pattern is ABSENT. This condition is met if the AP analysis fails to show distinct intratumoral arteries, OR if the PVP/DP analysis reveals the presence of a peritumoral hypodense halo.',
            'The combined pattern is PRESENT. This requires a two-step confirmation: 1) The AP analysis must show distinct, dot-like or linear hyper-enhancing structures (intratumoral arteries) inside the lesion, AND 2) The PVP/DP analysis must confirm the absence of a peritumoral hypodense halo.'
        ]
    },
    {
        "name": "Fade Enhancement Pattern", # 65
        "options": [
            'Comparing phases, the lesion becomes hypodense relative to surrounding liver in the PVP or DP, demonstrating a "washout" pattern.',
            'Comparing phases, the lesion enhancement persists, remaining iso- or hyper-enhancing relative to surrounding liver in the PVP or DP, demonstrating a "fade" (non-washout) pattern.'
        ]
    },
    {
        "name": "Nodule-in-Nodule Architecture", # 66
        "options": [
            'Across all phases, the lesion\'s internal enhancement is either homogeneous or chaotically heterogeneous, lacking a clear, stable hierarchical structure.',
            'A "nodule-in-nodule" architecture is confirmed across phases: a smaller nodule shows more intense AP enhancement than the larger parent lesion, and this distinction often persists in later phases.'
        ]
    },
    {
        "name": "Peripheral Washout", # 68
        "options": [
            'After initial AP enhancement, the lesion either does not show washout or shows a non-peripheral (diffuse) washout pattern in the PVP/DP.',
            'After initial AP enhancement, the lesion shows a distinct "peripheral washout" pattern, with only its rim becoming hypoenhancing in the PVP/DP.'
        ]
    },
    {
        "name": "Delayed Central Enhancement", # 70
        "options": [
            'Comparing phases, the central part of the lesion does not show progressive enhancement (e.g., it washes out or remains persistently non-enhancing).',
            'Comparing phases, the central part of the lesion shows progressive, sustained enhancement, becoming brighter in the delayed phase than it was in the AP.'
        ]
    }
]


# --- 2. 辅助函数 (Helper Functions) ---

def encode_image_to_base64(image_path):
    # (此函数用于将图片编码为Base64)
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        cprint(f"Warning: Image file not found at {image_path}. Skipping.", 'red')
        return None

# --- 3. 智能体核心类 (Agent Core Class) ---

class Agent:
    def __init__(self, instruction, role):
        # (此处的 Agent 类与我们之前版本相同，使用 Gemini API)
        self.system_instruction = {"parts": [{"text": instruction}]}
        self.role = role
        self.config = VERTEX_AI_CONFIG
        self.proxies = PROXY_CONFIG if PROXY_CONFIG.get('https') else None

    def _get_access_token(self):
        # (此函数用于通过服务账号获取访问令牌)
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.config['credentials_path'],
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            return credentials.token
        except Exception as e:
            cprint(f"Error authenticating from service account file: {e}", 'red')
            raise

    def chat(self, prompt_text, image_paths=None):
        # (此函数是 Agent 的核心，负责调用 Gemini API)
        if image_paths is None:
            image_paths = []
            
        try:
            access_token = self._get_access_token()
        except Exception:
            return "Could not authenticate. Please check your service account credentials path and permissions."

        url = f"https://{self.config['location']}-aiplatform.googleapis.com/v1/projects/{self.config['project_id']}/locations/{self.config['location']}/publishers/google/models/{self.config['model_name']}:streamGenerateContent"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        parts = [{"text": prompt_text}]
        valid_image_count = 0
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                parts.append({"inline_data": {"mime_type": "image/png", "data": base64_image}})
                valid_image_count += 1
        
        payload = {
            "system_instruction": self.system_instruction,
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192}
        }
        
        cprint(f" Agent '{self.role}' is sending request to {self.config['model_name']} with {valid_image_count} image(s)...", 'blue')

        try:
            response = requests.post(url, headers=headers, json=payload, proxies=self.proxies)
            if response.status_code != 200:
                cprint(f"API Error: {response.status_code}\nResponse: {response.text}", 'red')
                return f"API Error: Received status code {response.status_code}"
            
            response.raise_for_status()
            
            full_text_response = "".join(
                item['candidates'][0]['content']['parts'][0]['text']
                for item in response.json()
                if 'candidates' in item and 'content' in item['candidates'][0] and 'parts' in item['candidates'][0]['content']
            )
            
            if not full_text_response:
                cprint(f"Error: Response contained no text. Full Response: {response.json()}", 'red')
                return "Error: Model returned an empty response."

            return full_text_response

        except requests.exceptions.RequestException as e:
            cprint(f"Network error calling Vertex AI API: {e}", 'red')
            return f"Network Error: {e}"
        except Exception as e:
            cprint(f"An unknown error occurred while processing Vertex AI response: {e}", 'red')
            return f"Unknown Error: {e}"




# --- 4. 工作流各阶段函数 (Workflow Phase Functions) ---

def select_best_five_slices(image_paths_pvp):
    """
    使用两阶段AI流程，从一个期相的图片列表中智能选出一个包含5张图片的“代表性组合”。

    :param image_paths_pvp: PVP期相的图片路径列表。
    :return: 选出的5张图片的路径列表 (list)，如果失败则返回 None。
    """
    cprint("\n--- [启动“代表性切片组合”智能筛选子流程] ---", 'blue', attrs=['bold'])

    # --- 定义本流程所需的Agent和问题列表 ---

    # 1a. 切片评分大师 (Slice Scoring Master)
    scoring_master_instruction = """
    You are an expert radiologist. Your task is to evaluate a series of CT slices and score each one's **diagnostic utility**.
    You will be provided with multiple images. Your final output must be a single, valid JSON object containing a list of scorecards, one for each slice.
    """
    scoring_master_agent = Agent(instruction=scoring_master_instruction, role="Slice Scoring Master")

    # 1b. 甄选决策官 (Selection Judge)
    selection_judge_instruction = """
    You are the head of radiology. You have been provided with scorecards for multiple CT slices.
    Your task is to select the **5 most representative slices** that, as a group, provide the best overall view of the lesion and are most likely to help answer the clinical questions.
    **Do not simply pick the 5 highest scores.** Aim for a selection that includes the slice with the highest score, but also other slices that might show different aspects of the lesion (e.g., the top edge, the bottom edge, a different internal pattern). Your goal is **coverage and diversity**.
    Your final output must be a JSON object containing a list of the 5 chosen slice numbers.
    """
    selection_judge_agent = Agent(instruction=selection_judge_instruction, role="Selection Judge")

    # 1c. 任务所需分析的问题列表
    clinical_questions = [
        "1. Enhancing Capsule",
        "2. Peritumoral Perfusion Alteration",
        "3. Peritumoral Hypodense Halo",
        "4. Corona Enhancement",
        "5. Custom TTPVI Feature",
        "6. Fade Enhancement Pattern",
        "7. Nodule-in-Nodule Architecture",
        "8. Peripheral Washout",
        "9. Delayed Central Enhancement"
    ]
    
    # --- 阶段一：整体评分 (“海选”) ---
    cprint("--- [海选阶段] 正在将所有PVP切片一次性发送给评分大师进行比较和评分...", 'blue')
    
    slice_numbers = [int(re.search(r'(\d+)\.png$', path).group(1)) for path in image_paths_pvp if re.search(r'(\d+)\.png$', path)]

    scorer_task_prompt = f"""
    You have been provided with {len(image_paths_pvp)} CT images from a PVP sequence, numbered: {slice_numbers}.
    Your task is to **compare all of these images** and generate a scorecard for EACH slice.
    The scores should be relative to this specific set of images.

    For each slice, assign a **"Diagnostic Value Score"** from 1 (not useful) to 10 (critically important).
    A high score should be given to slices that are rich in features and crucial for answering the following clinical questions: {', '.join(clinical_questions)}.
    A slice that clearly shows multiple distinct features should get a very high score.

    Your output MUST be a single, valid JSON object with a root key "scorecards", containing a list of {len(image_paths_pvp)} scorecard objects.
    Each scorecard must have "slice_number", "diagnostic_value_score", and a "reason".
    
    REQUIRED JSON OUTPUT FORMAT:
    {{
        "scorecards": [
            {{
                "slice_number": [slice_num_1],
                "diagnostic_value_score": [Your score 1-10],
                "reason": "Briefly explain why this slice is diagnostically valuable."
            }},
            ...
        ]
    }}
    """
    
    response_text = scoring_master_agent.chat(prompt_text=scorer_task_prompt, image_paths=image_paths_pvp)

    scorecards = []
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        parsed_response = json.loads(clean_json_text)
        scorecards = parsed_response.get("scorecards", [])
        if scorecards:
            cprint("✅ 评分大师已成功返回评分报告。", 'green')
            print(f"Scorecards: \n{json.dumps(scorecards, indent=2)}\n")
    except (json.JSONDecodeError, AttributeError):
        cprint(f"Error: Failed to parse JSON from Slice Scoring Master. Response:\n{response_text}", 'red')

    # --- 阶段二：智能甄选 (“决选”) ---
    if not scorecards:
        cprint("Error: No valid scorecards were generated. Cannot select slices.", 'red')
        return None

    cprint("\n--- [决选阶段] 开始根据评分报告智能选择5张代表性切片...", 'blue')
    
    judge_task_prompt = f"""
    Here are the scorecards for {len(scorecards)} different CT slices. Please select the 5 best and most representative slice numbers.
    Your goal is to choose a diverse set that offers the best overall diagnostic coverage.

    **Scorecards:**
    {json.dumps(scorecards, indent=2)}

    **REQUIRED JSON OUTPUT FORMAT:**
    {{
        "selected_slice_numbers": [a, b, c, d, e]
    }}
    """
    
    response_text = selection_judge_agent.chat(prompt_text=judge_task_prompt)
    
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(clean_json_text)
        selected_numbers = decision["selected_slice_numbers"]
        
        if len(selected_numbers) != 5:
            cprint(f"Warning: Selection Judge returned {len(selected_numbers)} slices instead of 5. Using the result anyway.", 'yellow')

        cprint(f"✅ AI has selected the 5 most representative slice numbers: {selected_numbers}", 'green', attrs=['bold'])
        
        return selected_numbers

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        cprint(f"Error: Failed to parse JSON from Selection Judge. Error: {e}. Response:\n{response_text}", 'red')
        return None


def select_paths_by_numbers(
    selected_numbers: List[int], 
    image_paths_ap: List[str], 
    image_paths_dp: List[str], 
    image_paths_pvp: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    根据给定的切片号码列表，从三个期相的完整路径列表中筛选出对应的文件路径。

    :param selected_numbers: 一个包含AI选出的切片号码的列表, e.g., [18, 20, 25, 28, 30]。
    :param image_paths_ap: AP期相的完整图片路径列表。
    :param image_paths_dp: DP期相的完整图片路径列表。
    :param image_paths_pvp: PVP期相的完整图片路径列表。
    :return: 一个包含三个筛选后路径列表的元组 (selected_ap, selected_dp, selected_pvp)。
    """
    
    # --- 步骤一：构建“号码 -> 路径”的快速查找字典 ---
    # 这样做比每次都循环查找要高效得多。
    
    def create_path_map(paths: List[str]) -> dict:
        """辅助函数：从路径列表中创建号码到路径的映射。"""
        path_map = {}
        for path in paths:
            # 使用正则表达式从文件名中安全地提取数字
            match = re.search(r'(\d+)\.png$', path)
            if match:
                slice_num = int(match.group(1))
                path_map[slice_num] = path
        return path_map

    ap_map = create_path_map(image_paths_ap)
    dp_map = create_path_map(image_paths_dp)
    pvp_map = create_path_map(image_paths_pvp)

    # --- 步骤二：根据号码列表查找并筛选路径 ---
    # 使用 .get(num) 方法可以安全地处理号码不存在的情况（返回None）。
    # if ... is not None 确保了只有成功找到的路径才会被加入列表。
    
    selected_paths_ap = [ap_map.get(num) for num in selected_numbers if ap_map.get(num) is not None]
    selected_paths_dp = [dp_map.get(num) for num in selected_numbers if dp_map.get(num) is not None]
    selected_paths_pvp = [pvp_map.get(num) for num in selected_numbers if pvp_map.get(num) is not None]
    
    print(f"selected_paths_ap: {selected_paths_ap}")
    print(f"selected_paths_dp: {selected_paths_dp}")
    print(f"selected_paths_pvp: {selected_paths_pvp}")

    return selected_paths_ap, selected_paths_dp, selected_paths_pvp






# 分析3个时期的agent
def run_phase_1(image_paths_ap, image_paths_dp, image_paths_pvp):
    cprint("\n--- [Phase 1: Parallel Feature Extraction with Professional Questions] ---", 'yellow', attrs=['bold'])
    
    # 3. Agent初始化和调用 
    # 为动脉期(AP)分析师创建指令
    ap_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze single CT images from the **Arterial Phase (AP)**.
    Your analysis will be a structured JSON report. It is CRITICAL that your JSON is perfectly formatted and self-contained, as it will be the direct input for a senior AI analyst who will synthesize your report with reports from other phases. Their success depends on the accuracy and clarity of your report.
    You must answer all the specific questions provided below and format your output as a JSON object. 
    Do not add any explanatory text outside of the JSON structure.
    """

    # 为延迟期(DP)分析师创建指令
    dp_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze single CT images from the **Delayed Phase (DP)**.
    Your analysis will be a structured JSON report. It is CRITICAL that your JSON is perfectly formatted and self-contained, as it will be the direct input for a senior AI analyst who will synthesize your report with reports from other phases. Their success depends on the accuracy and clarity of your report.
    You must answer all the specific questions provided below and format your output as a JSON object. 
    Do not add any explanatory text outside of the JSON structure.
    """

    # 为门脉期(PVP)分析师创建指令
    pvp_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze single CT images from the **Portal Venous Phase (PVP)**.
    Your analysis will be a structured JSON report. It is CRITICAL that your JSON is perfectly formatted and self-contained, as it will be the direct input for a senior AI analyst who will synthesize your report with reports from other phases. Their success depends on the accuracy and clarity of your report.
    You must answer all the specific questions provided below and format your output as a JSON object. 
    Do not add any explanatory text outside of the JSON structure.
    """

    agent_1 = Agent(instruction=ap_analyst_instruction, role="phase AP Analyst")
    agent_2 = Agent(instruction=dp_analyst_instruction, role="phase DP Analyst")
    agent_3 = Agent(instruction=pvp_analyst_instruction, role="phase PVP Analyst")


    # 2. 准备存储所有结果的字典
    all_reports = {"AP": [], "DP": [], "PVP": []}

    # 3. 外层循环：遍历三个期相
    for phase, paths, agent in [("AP", image_paths_ap, agent_1), 
                                ("DP", image_paths_dp, agent_2), 
                                ("PVP", image_paths_pvp, agent_3)]:
        
        if not paths:
            cprint(f"No slices to process for phase {phase}. Skipping.", 'yellow')
            continue

        cprint(f"\n--- Processing Phase: {phase} ---", 'blue')
        
        # 4. 内层循环：遍历当前期相的每一张切片
        for image_path in tqdm(paths, desc=f"Analyzing {phase} Slices"):
            
            # 从文件名中提取切片号
            slice_number = "Unknown"
            match = re.search(r'(\d+)\.png$', image_path)
            if match:
                slice_number = match.group(1)

            # 5. 为当前期相筛选相关问题 (这部分逻辑不变)
            questions_for_this_phase = [
                f for f in FEATURE_DEFINITIONS if phase in FEATURE_PHASE_REQUIREMENTS.get(f['name'], [])
            ]

            if not questions_for_this_phase:
                continue # 如果当前期相没有需要回答的问题，则跳过

            # 6. 动态构建只包含相关问题和当前切片信息的高级Prompt
            prompt_sections = []
            json_findings_template = []
            for i, feature in enumerate(questions_for_this_phase):
                prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Option 0 (Feature Absent): "{feature['options'][0]}"
Option 1 (Feature Present): "{feature['options'][1]}"
""")
                json_findings_template.append(
                    f'{{"feature": "{feature["name"]}", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}}'
                )
            
            task_prompt = f"""
You are a meticulous radiologist analyzing a **single CT image** from the **{phase} phase (slice number: {slice_number})**.
For each of the {len(questions_for_this_phase)} features below, analyze the provided image and determine which option (0 or 1) is most accurate.
Format your final output as a SINGLE, VALID JSON object.

{''.join(prompt_sections)}

Your JSON output must follow this exact structure:
{{
  "phase": "{phase}",
  "slice_number": {slice_number},
  "findings": [
    {', '.join(json_findings_template)}
  ]
}}
"""
            # 7. 每次调用只输入1张图片
            response_text = agent.chat(prompt_text=task_prompt, image_paths=[image_path])
            
            # 8. 解析并存储该切片的报告
            try:
                clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
                report_for_slice = json.loads(clean_json_text)
                all_reports[phase].append(report_for_slice)
                cprint(f"  Successfully processed slice {slice_number} for phase {phase}.", 'green')
            except (json.JSONDecodeError, AttributeError):
                cprint(f"  Error: Failed to parse JSON for slice {slice_number} (Phase {phase}). Response:\n{response_text}", 'red')
                all_reports[phase].append({"error": "Failed to parse", "slice": slice_number, "raw_response": response_text})

    print(f"phase_1 output: {all_reports}")
    return all_reports


# 
def run_phase_2(phase1_reports, selected_slice_numbers):
    """
    第二阶段 (左分支): 跨期相文本整合。
    按切片号，为每个切片整合其在AP, DP, PVP三个期相的初级报告。
    这5个独立的整合任务将并行执行。
    """
    cprint("\n--- [Phase 2: Per-Slice Text Synthesis (Parallel)] ---", 'yellow', attrs=['bold'])

    # 1. 初始化 Agent
    # 这个Agent的角色是整合单一层面的三份报告
    text_synthesis_instruction = """
    You are a senior radiologist acting as a **cross-phase synthesis specialist**.
    Your task is to synthesize three phase-specific reports (AP, DP, PVP) that all describe the **same single anatomical slice**.
    Your goal is to create one cohesive, text-based analysis report that summarizes the lesion's complete dynamic behavior on this specific slice.
    Highlight the evolution of features, any inconsistencies, and the overall pattern observed.
    Your output should be a concise, analytical paragraph.
    """
    agent_4 = Agent(instruction=text_synthesis_instruction, role="Text Synthesis Analyst")
    
    # 2. 准备并行执行
    synthesized_reports_by_slice = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_slice = {}
        
        # 3. 外层循环：遍历5个被选中的切片号
        for slice_num in selected_slice_numbers:
            cprint(f"Submitting text synthesis task for slice #{slice_num}...", 'magenta')
            
            # 4. 为当前切片号，从phase1_reports中收集三份对应的报告
            try:
                # 使用列表推导式高效地找到对应切片的报告
                report_ap = next(item for item in phase1_reports['AP'] if item.get('slice_number') == slice_num)
                report_dp = next(item for item in phase1_reports['DP'] if item.get('slice_number') == slice_num)
                report_pvp = next(item for item in phase1_reports['PVP'] if item.get('slice_number') == slice_num)
            except StopIteration:
                cprint(f"Warning: Missing one or more phase reports for slice #{slice_num}. Skipping.", 'yellow')
                continue

            # 5. 构建专属的、关于当前切片的Prompt
            task_prompt = f"""
            Please synthesize the following three reports for **slice number {slice_num}** into a single, comprehensive text analysis.

            [Report from phase AP for slice {slice_num}]:
            {json.dumps(report_ap.get('findings', []), indent=2, ensure_ascii=False)}

            [Report from phase DP for slice {slice_num}]:
            {json.dumps(report_dp.get('findings', []), indent=2, ensure_ascii=False)}

            [Report from phase PVP for slice {slice_num}]:
            {json.dumps(report_pvp.get('findings', []), indent=2, ensure_ascii=False)}

            Your output should be a single, analytical paragraph summarizing your findings for this slice.
            """
            
            # 6. 提交并行任务
            future = executor.submit(agent_4.chat, prompt_text=task_prompt)
            future_to_slice[future] = slice_num

        # 7. 收集并行任务的结果
        cprint("Waiting for per-slice text synthesis tasks to complete...", 'blue')
        for future in tqdm(as_completed(future_to_slice), total=len(future_to_slice), desc="Synthesizing Slices"):
            slice_num = future_to_slice[future]
            try:
                result_text = future.result()
                synthesized_reports_by_slice[slice_num] = result_text
                synthesized_reports_by_slice = dict(sorted(synthesized_reports_by_slice.items()))
            except Exception as exc:
                cprint(f"\nAn exception occurred while synthesizing slice #{slice_num}: {exc}", 'red')
                synthesized_reports_by_slice[slice_num] = {"error": str(exc)}

    cprint("\n--- Text Synthesis Reports (by Slice) ---", 'green')
    print(json.dumps(synthesized_reports_by_slice, indent=2, ensure_ascii=False))

    return synthesized_reports_by_slice


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def _analyze_single_slice_level_visual(agent, slice_num, focus_image_paths):
    """
    【修正版】辅助函数：对单一解剖层面对应的3张期相图片进行详细视觉分析。
    """
    
    # 动态构建问题列表 (这部分逻辑不变)
    prompt_sections = []
    for i, feature in enumerate(FEATURE_DEFINITIONS):
        prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Answer 0 (Absent): "{feature['options'][0]}"
Answer 1 (Present): "{feature['options'][1]}"
""")

    # 【修正版】Prompt现在只关注这3张核心图片
    fine_grained_visual_prompt = f"""
You are an expert radiologist, the **Visual Adjudicator** for an AI committee. Your task is to perform a fine-grained analysis of a **single anatomical slice level** across its three contrast phases.

**IMAGE CONTEXT:**
You have been provided with **3 CT images** corresponding to slice number **{slice_num}**:
- Image 1: Arterial Phase (AP)
- Image 2: Delayed Phase (DP)
- Image 3: Portal Venous Phase (PVP)

**YOUR TASK:**
Synthesize the visual information for slice **{slice_num}**. For each of the {len(FEATURE_DEFINITIONS)} features, choose the most accurate description (0 or 1), provide a confidence score (1-5), and a justification. Your justification MUST be based on direct, multi-phase visual evidence from the provided images for this slice level.

**FEW-SHOT EXAMPLES:**
Here are examples of the high-quality, evidence-based reasoning required:
{{
    "pattern_1": {{
        "answer": 1,
        "confidence": 5,
        "justification": "On slice #{slice_num}, review of the Delayed Phase (DP) sequence (images 6-10) reveals a distinct and smooth hyper-enhancing rim completely encircling the lesion, which was not visible in the Arterial Phase. This feature is consistent with the adjacent slices."
    }},
    "pattern_2": {{
        "answer": 0,
        "confidence": 4,
        "justification": "On slice #{slice_num}, review of the Arterial Phase (AP) sequence (images 1-5) shows homogeneous enhancement in the liver parenchyma surrounding the lesion, with no evidence of the characteristic wedge-shaped or halo-like hyperenhancement, and this observation is consistent with the adjacent slices."
    }}
}}

**QUESTIONS TO ANSWER for Slice #{slice_num}:**
{''.join(prompt_sections)}

**REQUIRED OUTPUT FORMAT:**
Your output MUST be a SINGLE, VALID JSON object for this slice. It should contain a list of {len(FEATURE_DEFINITIONS)} findings.
Example: {{ "slice_findings": [...] }}
"""
    
    # 【【核心修正】】
    # 工人函数现在只将【3张焦点图片】的路径传递给API
    response_text = agent.chat(prompt_text=fine_grained_visual_prompt, image_paths=focus_image_paths)
    
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json_text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON", "raw_response": response_text, "slice_number": slice_num}
    

def run_phase_visual(selected_ap, selected_dp, selected_pvp, selected_numbers):
    cprint("\n--- [Visual Adjudicator: Fine-grained, Per-Slice Analysis (Parallel)] ---", 'yellow', attrs=['bold'])

    visual_adjudicator_instruction = "You are an expert radiologist, the Visual Adjudicator for an AI committee. Provide a detailed report for one specific slice at a time based on its 3-phase images. Follow all instructions in the user prompt precisely."
    agent_5 = Agent(instruction=visual_adjudicator_instruction, role="Visual Adjudicator")
    
    final_visual_reports = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_slice = {}
        
        ap_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_ap}
        dp_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_dp}
        pvp_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_pvp}

        cprint(f"Submitting {len(selected_numbers)} fine-grained visual analysis tasks to be run in parallel...", 'magenta')
        for slice_num in selected_numbers:
            # 为当前切片号，明确地收集其对应的3张焦点图片
            focus_images_for_level = [ap_map.get(slice_num), dp_map.get(slice_num), pvp_map.get(slice_num)]
            
            if all(focus_images_for_level):
                # 【【核心修正】】
                # “经理”现在将正确的参数（3张焦点图片）委托给“工人”函数
                future = executor.submit(
                    _analyze_single_slice_level_visual, 
                    agent_5, 
                    slice_num, 
                    focus_images_for_level # <-- 现在正确地传递了3张焦点图片
                )
                future_to_slice[future] = slice_num

        # ... (后续的tqdm和结果收集逻辑不变) ...
        for future in tqdm(as_completed(future_to_slice), total=len(future_to_slice), desc="Adjudicating Slices"):
            slice_num = future_to_slice[future]
            result = future.result()
            final_visual_reports[slice_num] = result
            
    cprint("\n--- Fine-Grained Visual Adjudicator Output (by Slice) ---", 'green')
    print(json.dumps(final_visual_reports, indent=2, ensure_ascii=False))
    
    return final_visual_reports
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————







def run_phase_3(longitudinal_report, cross_sectional_report, visual_adjudicator_report):
    """
    执行第三阶段：运行首席整合官，综合所有分析报告。
    """
    cprint("👨‍⚕️ [Phase 3: Final Decision Making] 👨‍⚕️", 'cyan', attrs=['bold'])
    
    # 准备给首席整合官的输入，现在包含三份报告
    synthesis_input = f"""
    [Feature Evolution Report from Cross-Phase Analyst]:
    {longitudinal_report}
    [Diagnostic Snapshot Report from Pattern Recognition Analyst]:
    {cross_sectional_report}
    [Visual Adjudicator Report from Direct Image Analysis]:
    {visual_adjudicator_report}
    """
    cprint("\n[6. Chief Synthesizer] Synthesizing all three expert reports...", 'magenta')

    # Agent 5: 首席整合官
    chief_synthesizer_prompt = f"""
You are the **Chief Radiologist** presiding over an AI diagnostic committee. Your task is to provide the final, global conclusion by synthesizing three expert reports.

**Your Inputs:**
1.  A **Feature Evolution Report** from the Cross-Phase Analyst (summarizing text-based findings over time).
2.  A **Diagnostic Snapshot Report** from the Pattern Recognition Analyst (summarizing text-based findings by phase).
3.  A **Visual Adjudicator Report** (based on direct analysis of all images, with confidence scores).

**Your Reasoning Process:**
Your primary job is to synthesize three expert perspectives to form a final, reasoned conclusion. Based on testing, the text-based analysis (reports 1 & 2) has been found to be more consistently reliable. Therefore, you must follow this **Conflict Resolution Protocol**:

1.  **Establish Primary Finding from Text:** First, for each of the 9 features, establish a "primary finding" by synthesizing the **Feature Evolution Report** and the **Diagnostic Snapshot Report** (reports 1 & 2). This text-based conclusion is your baseline.

2.  **Use Visual Report for Verification:** Next, use the **Visual Adjudicator Report** (report 3) to either **confirm** or **challenge** this primary finding.

3.  **Handle Agreement and Conflict:**
    * **If they AGREE:** The finding is confirmed with high confidence. Your justification should reflect this consensus.
    * **If they CONFLICT:** This indicates a significant discrepancy. You must handle it with caution:
        * **Default Stance:** Your default decision should be to **trust the primary finding from the text-based analysis (reports 1 & 2)**.
        * **Condition for Override:** You may only override the text-based finding and adopt the Visual Adjudicator's conclusion if, and only if, the Visual Adjudicator's report meets **BOTH** of these strict criteria:
            a) Its confidence score is the **maximum possible (e.g., 5/5)**.
            b) Its justification provides **exceptionally clear, unambiguous, and compelling evidence**, citing specific image numbers.
        * If the override condition is not met, you stick with the text-based finding.

4.  **Justify Your Final Decision:** In your final `justification` for each feature, you must be transparent about the process:
    * If the reports agreed, state the consensus. (e.g., *"Confirmed by both text-based analysis and direct visual review."*)
    * If there was a conflict that you resolved, explain your decision. (e.g., *"While the Visual Adjudicator reported a possible finding, the text-based analysis from all initial phase-analysts was consistently negative. Defaulting to the more reliable text-based consensus."* or *"Overriding the text-based analysis due to a definitive, max-confidence (5/5) visual findin

Your final output MUST be a single block of text that strictly follows the three-part structure outlined below. It is CRITICAL that you include the exact headings for each section, including the numbering.
**Required Output Format:**
1.  **Core Conclusion:** 
    A concise paragraph summarizing the overall clinical findings and conclusion.

2.  **Main Evidence:**
    A bulleted points detailing the key evidence from both the longitudinal, cross-sectional and visual reports that support your core conclusion.

3.  **Structured Summary:**
    This section MUST be a single, valid JSON object and nothing else. Do not add any introductory text, markdown tags like ```json, or any text after the JSON object.
    The JSON object must summarize the final answer for each of the 9 features. It must have keys from "pattern_1" to "pattern_9", corresponding to the features in this order:
    1. Enhancing Capsule
    2. Peritumoral Perfusion Alteration
    3. Peritumoral Hypodense Halo
    4. Corona Enhancement
    5. Custom TTPVI Feature
    6. Fade Enhancement Pattern
    7. Nodule-in-Nodule Architecture
    8. Peripheral Washout
    9. Delayed Central Enhancement

    For each pattern, the value must be an object with two keys:
    - "answer": The final binary conclusion (0 for first description, 1 for second description).
    - "justification": A brief, concise summary of the reasoning.

**Before generating the final output, double-check that the JSON syntax is perfect, especially for **Structured Summary:**, ensuring that a comma (`,`) separates every pattern object from the next.**

Example of the required structure for the JSON part ONLY:
```json
{{
    "pattern_1": {{
        "answer": 1,
        "justification": "Visible as a distinct, enhancing rim in the PVP/DP, which was not clearly defined in the AP."
    }},
    "pattern_2": {{
        "answer": 0,
        "justification": "No transient, wedge-shaped hyperenhancement was observed; any minor surrounding brightness in the AP persisted into the PVP."
    }},
    "pattern_7": {{
        "answer": 0,
        "justification": "The lesion's internal enhancement was reported as homogeneous across all three phases, lacking a distinct inner nodule."
    }}
}}
"""

    agent_6 = Agent(instruction=chief_synthesizer_prompt, role="Chief Synthesizer")
    final_hybrid_output = agent_6.chat(prompt_text=synthesis_input)
    # print(f"agent_6 output: \n{final_hybrid_output}")


    # 【【新】】增加解析逻辑，分离文本报告和JSON对象
    prose_report = ""

    try:
        # 使用正则表达式来分割，更强大、更宽容
        pattern = re.compile(r'3\.\s*\**Structured\s+Summary\**:', re.IGNORECASE)
        parts = pattern.split(final_hybrid_output, 1)

        if len(parts) == 2:
            prose_report = parts[0].strip()
            json_text = parts[1].strip() # 这是可能包含杂质的JSON文本
            
            # 【【【核心修改：智能提取】】】
            # 从可能混杂的文本中，精确找到第一个 '{' 和最后一个 '}'
            start_index = json_text.find('{')
            end_index = json_text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                # 精确提取从 '{' 到 '}' 的所有内容
                actual_json_text = json_text[start_index : end_index + 1]
                
                # 现在对这个纯净的字符串进行解析
                structured_summary = json.loads(actual_json_text)
                cprint("✅ Successfully extracted and parsed JSON.", 'green')
            else:
                # 如果连 '{' 和 '}' 都找不到
                cprint("Error: Could not find a valid JSON object within the summary section.", "red")
                structured_summary = {"error": "JSON object start '{' or end '}' not found.", "raw_json_text": json_text}
        else:
            # 如果正则表达式没有找到匹配的分割点
            prose_report = final_hybrid_output
            structured_summary = {"error": "Structured Summary section heading not found in the output.", "raw_response": final_hybrid_output}

    except Exception as e:
        cprint(f"An unexpected error occurred during parsing: {e}", "red")
        prose_report = final_hybrid_output
        structured_summary = {"error": f"An unexpected error occurred. Details: {e}", "raw_response": final_hybrid_output}

    # --- 打印最终结果 ---
    print("\n" + "="*20 + " 分割结果 " + "="*20)

    print("\n--- 第一部分：文本报告 (Prose Report) ---")
    print(prose_report)

    print("\n--- 第二部分：结构化总结 (Structured Summary JSON) ---")
    # 使用json.dumps美化打印输出，方便查看
    print(json.dumps(structured_summary, indent=4, ensure_ascii=False))

    # 返回两个部分：文本报告和结构化字典
    return prose_report, structured_summary
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


















# --- 5. 主执行逻辑 (Main Execution Logic) ---

def main():
    # (这是主函数，负责编排整个流程)
    all_patient_results = {}
    all_selected_paths = {}

    try:
        patient_ids = [d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
        patient_ids.sort()
        if not patient_ids:
            cprint(f"Error: No patient folders found in '{BASE_DATA_PATH}'.", 'red')
            return
        cprint(f"Found {len(patient_ids)} patient folder(s). Starting processing...", 'cyan', attrs=['bold'])
    except FileNotFoundError:
        cprint(f"Error: The directory '{BASE_DATA_PATH}' was not found. Please check the path.", 'red')
        return

    for patient_id in tqdm(patient_ids, desc="Processed Patients"):
        cprint(f"\n{'='*20} Processing Patient: {patient_id} {'='*20}", 'yellow')
        patient_folder = os.path.join(BASE_DATA_PATH, patient_id)
        
        # 为每个时期收集图像路径
        image_paths_ap = sorted([os.path.join(patient_folder, 'AP', f) for f in os.listdir(os.path.join(patient_folder, 'AP')) if f.endswith('.png')])
        image_paths_dp = sorted([os.path.join(patient_folder, 'DP', f) for f in os.listdir(os.path.join(patient_folder, 'DP')) if f.endswith('.png')])
        image_paths_pvp = sorted([os.path.join(patient_folder, 'PVP', f) for f in os.listdir(os.path.join(patient_folder, 'PVP')) if f.endswith('.png')])
        
        if not all([image_paths_ap, image_paths_dp, image_paths_pvp]):
            cprint(f"Warning: Patient {patient_id} is missing images in one or more phase folders. Skipping.", 'red')
            continue

        # 1. 【【新】】调用AI来寻找PVP期相的5张最佳代表图片
        selected_numbers = select_best_five_slices(image_paths_pvp)

        if selected_numbers is None:
            cprint(f"Could not select representative slices for patient {patient_id}. Skipping.", 'red')
            all_selected_paths[patient_id] = {"error": "Slice selection failed."}
            continue

        # 找出这5张CT图的路径合集
        selected_ap, selected_dp, selected_pvp = select_paths_by_numbers(
            selected_numbers,
            image_paths_ap,
            image_paths_dp,
            image_paths_pvp
        )
    
        # 3. 保存每个时期挑选的5个图片的path
        all_selected_paths[patient_id] = {
            "AP": selected_ap,
            "DP": selected_dp,
            "PVP": selected_pvp
        }
        cprint(f"Stored selected paths for patient {patient_id}.", "blue")





        # 执行第一阶段，每时期10张图
        phase1_report = run_phase_1(selected_ap, selected_dp, selected_pvp)
        # print(f"phase1 reports: {phase1_reports}")
        
        # 2. 执行第二阶段
        # all_images = selected_paths_ap + selected_paths_dp + selected_paths_pvp
        phase2_report= run_phase_2(phase1_report, selected_numbers)

        # 每时期3张图
        visual_report = run_phase_visual(selected_ap, selected_dp, selected_pvp, selected_numbers)
        
        # 3. 执行第三阶段
        # prose_report, structured_summary = run_phase_3(long_report, cross_report, visual_report)
        
        # 收集结果
        # all_patient_results[patient_id] = structured_summary







    # 将所有病人挑选的5个图片的path保存到一个JSON文件中
    output_filename = "selected_image_paths.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_selected_paths, f, ensure_ascii=False, indent=4)
        cprint(f"\n\n✅ Successfully saved all selected image paths to '{output_filename}'.", 'green', attrs=['bold'])
    except Exception as e:
        cprint(f"\n\n❌ Failed to save selected paths. Error: {e}", 'red')


    # 将所有病人的结果保存到一个JSON文件中
    output_filename = "selected_numbers.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_patient_results, f, ensure_ascii=False, indent=4)
    
    cprint(f"\n\n🎉 All processing complete. All reports have been saved to '{output_filename}'.", 'cyan', attrs=['bold'])


if __name__ == '__main__':
    main()
