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
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from typing import List, Tuple


# --- 1. 配置区 (Configuration Section) ---
# 请在此处修改您的个人配置

# 项目和模型配置
VERTEX_AI_CONFIG = {
    "project_id": "fm-bridge-agent",
    "location": "us-central1",
    # 【【【重要】】】请务必修改为您的服务账号密钥文件的真实路径
    "credentials_path": "/home/yxcui/FM-Bridge/testing_file/fm-bridge-agent-ba83686905f8.json",
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
    
    slice_numbers = [re.search(r'(\d+)\.png$', path).group(1) for path in image_paths_pvp if re.search(r'(\d+)\.png$', path)]
    scorer_task_prompt = f"""
You are an expert radiologist. Your task is to evaluate a series of CT slices from a single patient to score each slice's **diagnostic utility**.
**Your most important task is to be honest. Do not invent findings that are not clearly visible. It is a critical error to assign a high score to a slice with no diagnostic value.**

**Zero-Score Rule:**
**You MUST assign a "Diagnostic Value Score" of 0 or 1** for any slice that contains no discernible lesion, is outside the lesion's boundary, or provides no useful information for answering the clinical questions.

**Your Task:**
You have been provided with {len(image_paths_pvp)} CT images showing a **liver tumor** from a **Portal Venous Phase (PVP)** sequence. The slice numbers are: {slice_numbers}.
Your task is to compare all of these images and generate a scorecard for EACH slice.
A high score should be given to slices that are rich in features and crucial for answering the following clinical questions: {', '.join(clinical_questions)}.
For each slice, assign a **"Diagnostic Value Score"** from 1 (not useful) to 10 (critically important).
A slice that clearly shows multiple distinct features should get a very high score.

**Example for a HIGH-value slice and a ZERO-value slice:**
"scorecards": 
[
    {{
        "slice_number": 08,
        "diagnostic_value_score": 9,
        "reason": "Critically important slice demonstrating a classic, thick, enhancing capsule and internal washout, which are key features for diagnosis."
    }},
    {{
        "slice_number": 32,
        "diagnostic_value_score": 0,
        "reason": "This slice is outside the boundaries of the lesion and contains no diagnostic information. Its value is zero."
    }}
]     

**REQUIRED OUTPUT FORMAT:**
Your output MUST be a single, valid JSON object with a root key "scorecards", containing a list of {len(image_paths_pvp)} scorecard objects.
Each scorecard must have "slice_number", "diagnostic_value_score", and a "reason".
"""
    
    response_text = scoring_master_agent.chat(prompt_text=scorer_task_prompt, image_paths=image_paths_pvp)

    clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
    scorecards = clean_json_text

    if scorecards:
        cprint("✅ 评分大师已成功返回评分报告。", 'green')
        print(f"{scorecards}")

    # --- 阶段二：智能甄选 (“决选”) ---
    if not scorecards:
        cprint("Error: No valid scorecards were generated. Cannot select slices.", 'red')
        return None

    cprint("\n--- [决选阶段] 开始根据评分报告智能选择5张代表性切片...", 'blue')
    
    judge_task_prompt = f"""
You are the head of radiology. You have been provided with scorecards for 10 CT slices. 
Your task is to select an **optimal portfolio of 5 slices** that best tells the complete diagnostic story of the lesion.

You MUST follow this **Zonal Selection Strategy**:

**Step 1: Identify the "Peak Slices".**
First, read all scorecards to identify the 'Peak Slices'. These are the slices with the absolute highest scores (e.g., 9 or 10) that, according to their "reason" text, demonstrate the most critical diagnostic features (like a definitive capsule, clear washout, TTPVI, etc.).

**Step 2: Identify the "Core Body Slices".**
Next, identify the 'Core Body Slices'. This is typically a group of 2-3 slices with consecutively high scores (e.g., 7 or 8) that the "reason" text describes as showing the lesion's maximum diameter or main body.

**Step 3: Build the Final Portfolio.**
Construct your final list of 5 slices by strategically selecting from the groups you identified above. Your portfolio MUST include:
a) The single slice with the absolute highest score (the "Primary Peak").
b) At least one other "Peak Slice" that shows a different, critical diagnostic feature.
c) At least one "Core Body Slice" to represent the lesion's overall size and morphology.
d) Fill the remaining slots to ensure the best overall coverage and context.

**Final Task:**
Based on this zonal analysis, provide your final list of 5 selected slice numbers, sorted numerically.

**Scorecards:**
{scorecards}

**REQUIRED JSON OUTPUT FORMAT:**
{{
    "reasoning": "A brief summary of your selection. Example: 'Selected #37 as the primary peak for its clear capsule/washout, #44 for definitive washout, #28 and #30 to represent the core body, and #35 as a transitional slice.'",
    "selected_slice_numbers": [a, b, c, d, e]
}}
"""
    
    response_text = selection_judge_agent.chat(prompt_text=judge_task_prompt)
    
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(clean_json_text)
        raw_numbers = decision["selected_slice_numbers"]
        selected_numbers = [str(num).zfill(2) for num in raw_numbers]
        
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

# ----------------------------------------------------------------------------
def process_phase1_parallel(phase, paths, agent):
    """
    Analyzes all slices for a single phase and returns the collected reports.
    This is the function that will be executed in parallel by each thread.
    """
    if not paths:
        cprint(f"No slices to process for phase {phase}. Skipping.", 'yellow')
        return phase, []

    cprint(f"--- Starting parallel processing for Phase: {phase} ---", 'blue')
    
    phase_reports = []
    # Loop through each slice for this specific phase
    for image_path in tqdm(paths, desc=f"Analyzing {phase} Slices"):
        slice_number = "Unknown"
        match = re.search(r'(\d+)\.png$', image_path)
        if match:
            slice_number = match.group(1)

        # (The logic for building the prompt is the same as your original code)
        questions_for_this_phase = [
            f for f in FEATURE_DEFINITIONS if phase in FEATURE_PHASE_REQUIREMENTS.get(f['name'], [])
        ]
        if not questions_for_this_phase:
            continue

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
        # Call the agent for the single image
        response_text = agent.chat(prompt_text=task_prompt, image_paths=[image_path])
        
        # Parse and store the report
        try:
            clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
            report_for_slice = json.loads(clean_json_text)
            phase_reports.append(report_for_slice)
        except (json.JSONDecodeError, AttributeError):
            cprint(f"  Error: Failed to parse JSON for slice {slice_number} (Phase {phase}).", 'red')
            phase_reports.append({"error": "Failed to parse", "slice": slice_number, "raw_response": response_text})
            
    return phase, phase_reports

# --- Step 2: Rewrite the main run_phase_1 function to manage the parallel workers ---
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

    # We still create the three specialized agents
    agent_1 = Agent(instruction=ap_analyst_instruction, role="phase AP Analyst")
    agent_2 = Agent(instruction=dp_analyst_instruction, role="phase DP Analyst")
    agent_3 = Agent(instruction=pvp_analyst_instruction, role="phase PVP Analyst")
    
    # This dictionary will store the final results
    all_reports = {"AP": [], "DP": [], "PVP": []}
    
    # A list of all tasks to be run
    tasks = [
        ("AP", image_paths_ap, agent_1),
        ("DP", image_paths_dp, agent_2),
        ("PVP", image_paths_pvp, agent_3)
    ]

    # Use ThreadPoolExecutor to run the tasks in parallel
    # We set max_workers=3 because we have exactly 3 tasks to run concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit each phase's processing task to the thread pool
        future_to_phase = {executor.submit(process_phase1_parallel, phase, paths, agent): phase for phase, paths, agent in tasks}
        
        cprint(f"Submitted {len(future_to_phase)} phase analysis tasks to run in parallel.", 'magenta')

        # Wait for futures to complete and collect results
        for future in concurrent.futures.as_completed(future_to_phase):
            phase = future_to_phase[future]
            try:
                # Get the result from the completed task
                completed_phase, reports = future.result()
                all_reports[completed_phase] = reports
                cprint(f"✅ Finished processing all slices for Phase: {completed_phase}.", 'green')
            except Exception as exc:
                cprint(f"❌ Phase {phase} generated an exception: {exc}", 'red')

    print("phase_1 output (from parallel execution)")
    print(json.dumps(all_reports, indent=2, ensure_ascii=False))
    return all_reports


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
            Please synthesize the following three reports for **slice number {slice_num}**.
            For any given feature, you may only have input from one, two, or all three reports; this is expected. You must deduce the overall evolution based on the available information. For example, to determine "Peritumoral Perfusion Alteration", you should primarily rely on the AP and PVP reports. 
            
            **YOUR PRIMARY TASK:**
            Create a definitive summary for ALL 9 of the following radiological features**. You must address each feature one by one, synthesizing the findings from the provided reports to determine the overall conclusion for this slice.

            **YOUR ROLE:**
            Your role is to **accurately consolidate the provided information. Do not introduce new visual findings that are not in the reports. Your goal is to create a unified text summary for each feature.
            
            [Report from phase AP for slice {slice_num}]:
            {json.dumps(report_ap.get('findings', []), indent=2, ensure_ascii=False)}

            [Report from phase DP for slice {slice_num}]:
            {json.dumps(report_dp.get('findings', []), indent=2, ensure_ascii=False)}

            [Report from phase PVP for slice {slice_num}]:
            {json.dumps(report_pvp.get('findings', []), indent=2, ensure_ascii=False)}

            **REQUIRED OUTPUT FORMAT:**
            Your output MUST be a point-by-point summary. Use the exact feature names as headings. Do not use a single narrative paragraph.

            ---
            **REQUIRED OUTPUT FORMAT EXAMPLE:**
            **Enhancing Capsule:** [Your synthesized finding for this feature, e.g., "Absent in the AP report, but described as a clear, smooth capsule in both the PVP and DP reports, confirming its presence."]
            **Peritumoral Perfusion Alteration:** [Your synthesized finding for this feature, e.g., "The AP report noted a transient hyperenhancement that was confirmed to resolve in the PVP report, indicating the feature is present."]
            **Peritumoral Hypodense Halo:** [Your synthesized finding for this feature...]
            ---

            Now, generate the complete, point-by-point summary for all 9 features for slice #{slice_num}.
            """
            # print(f"long agent input for slice {slice_num}: {task_prompt}")
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
# 视觉分析师，在run_phase_3中使用
def _analyze_single_slice_visual_window(agent_visual, slice_num, images_for_window, prev_slice_num, next_slice_num):
    #【升级版】辅助函数：在一个“视觉窗口”内，对中心切片进行分析。
    # 动态构建问题列表 (这部分逻辑不变)
    prompt_sections = []
    for i, feature in enumerate(FEATURE_DEFINITIONS):
        prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Answer 0 (Absent): "{feature['options'][0]}"
Answer 1 (Present): "{feature['options'][1]}"
""")
    # 1. 构建一个详细的、带编号的图片清单字符串
    image_context_parts = []
    focus_start_index = 1
    if prev_slice_num is not None:
        image_context_parts.append(f"- **Context (Neighbor #{prev_slice_num}):** Images 1-3 (AP, DP, PVP)")
        focus_start_index = 4
    if next_slice_num is not None:
        context_next_start_index = focus_start_index + 3
        image_context_parts.append(f"- **Context (Neighbor #{next_slice_num}):** Images {context_next_start_index}-{context_next_start_index+2} (AP, DP, PVP)")

    # This line always comes first
    image_context_parts.insert(0, f"- **Primary Focus (Slice #{slice_num}):** Images {focus_start_index}-{focus_start_index+2} (AP, DP, PVP)")

    image_context_str = "\n".join(image_context_parts)
    # print(f"告诉agent图片的顺序: \n{image_context_str}")

    # 构建新的、完全动态的Prompt
    visual_window_prompt = f"""
You are an expert radiologist, the **Visual Adjudicator** for an AI committee. Your judgment is final as it is based on direct visual evidence.

### INPUT OVERVIEW
You have been provided with a "visual window" of {len(images_for_window)} CT images to analyze.
Here is the manifest detailing each image you have received:
{image_context_str}

### PRIMARY OBJECTIVE
Your goal is to produce a detailed, fine-grained analysis report for the **single central slice** within this window, which is **slice number {slice_num}**. You will use the adjacent slices as 3D context.
Be extremely conservative; start with a baseline of 3 and only increase if the evidence is exceptionally strong.

### STEP-BY-STEP INSTRUCTIONS
1.  **Build 3D Context:** First, review all provided images using the manifest to form a three-dimensional understanding of the lesion's structure.
2.  **Focus on the Central Slice:** Now, focus your primary analysis on the 3 phase images (AP, DP, PVP) for **slice #{slice_num}**.
3.  **Answer All Questions:** For each of the {len(FEATURE_DEFINITIONS)} features listed below, you must:
    a. Choose the most accurate description (0 or 1) for slice #{slice_num}.
    b. Provide a confidence score (1-5).
    c. Write a justification. Your justification **must** primarily describe the findings on slice #{slice_num}, but **should also** state whether these findings are consistent with the adjacent slices (your 3D context).
4.  **Format the Output:** Construct a SINGLE, VALID JSON object as described below.

### QUESTIONS TO ANSWER for Slice #{slice_num}
{''.join(prompt_sections)}

### REQUIRED OUTPUT FORMAT & EXAMPLE
Your final output MUST be a SINGLE, VALID JSON object and nothing else. The root of the object should contain {len(FEATURE_DEFINITIONS)} keys, named sequentially from "pattern_1" to "pattern_{len(FEATURE_DEFINITIONS)}". Before you output, double-check that your JSON is perfectly formatted.
**FEW-SHOT EXAMPLES:**
Here are examples of the high-quality, evidence-based reasoning required:
```json
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
"""
    
    # 【【核心修正】】
    # 工人函数现在只将【3张焦点图片】的路径传递给API
    response_text = agent_visual.chat(prompt_text=visual_window_prompt, image_paths=images_for_window)
    
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json_text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON", "raw_response": response_text, "slice_number": slice_num}

def run_phase_3(selected_ap, selected_dp, selected_pvp, selected_numbers):
    cprint("\n--- [Visual Adjudicator: Fine-grained Analysis with Visual Window (Parallel)] ---", 'yellow', attrs=['bold'])

    # ... (Agent初始化和path_map构建部分保持不变) ...
    visual_adjudicator_instruction = "You are an expert radiologist, the Visual Adjudicator for an AI committee. Provide a detailed report for one specific slice at a time based on its 3-phase images. Follow all instructions in the user prompt precisely."
    agent_5 = Agent(instruction=visual_adjudicator_instruction, role="Visual Adjudicator")
    ap_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_ap}
    dp_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_dp}
    pvp_map = {int(re.search(r'(\d+)\.png$', p).group(1)): p for p in selected_pvp}
    
    final_visual_reports = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_slice = {}
        
        # 排序以确保索引的正确性
        sorted_selected_numbers = sorted(selected_numbers)

        for i, slice_num in enumerate(sorted_selected_numbers):
            
            # 【【核心逻辑】】确定邻居切片的号码
            prev_slice_num = sorted_selected_numbers[i-1] if i > 0 else None
            next_slice_num = sorted_selected_numbers[i+1] if i < len(sorted_selected_numbers) - 1 else None

            # 收集当前切片和其邻居的图片 (这个逻辑是正确的)
            images_for_window = []
            for num in [prev_slice_num, slice_num, next_slice_num]: # 哪3个slice
                if num is not None:
                    images_for_window.extend([ap_map.get(num), dp_map.get(num), pvp_map.get(num)])
            images_for_window = [p for p in images_for_window if p is not None] # 最后只有一个list

            if len(images_for_window) < 3:
                continue
            
            # 【【核心修改】】将邻居信息也一并提交给“工人”
            # 检查过了，3D输入是对的，3张切片，6或9张图片 ！！！
            future = executor.submit(
                _analyze_single_slice_visual_window, 
                agent_5, 
                slice_num, 
                images_for_window, # 一个list包含2或3个切片的3个时期图像(6或9张图片)
                prev_slice_num, # 前一张CT图的号码
                next_slice_num  # 后一张CT图的号码
            )
            future_to_slice[future] = slice_num

        for future in tqdm(as_completed(future_to_slice), total=len(future_to_slice), desc="Adjudicating Slices"):
            slice_num = future_to_slice[future]
            result = future.result()
            final_visual_reports[slice_num] = result

        final_visual_reports = dict(sorted(final_visual_reports.items(), key=lambda item: int(item[0])))

            
    cprint("\n--- Fine-Grained Visual Adjudicator Output (by Slice) ---", 'green')
    print(json.dumps(final_visual_reports, indent=2, ensure_ascii=False))

    return final_visual_reports
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————





# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# chief
# 负责输入单个切片给agent的函数，在run_phase_4中使用
def _adjudicate_single_slice(agent_6, slice_num, text_report, visual_report):
    """
    对单一解剖层面，综合其文本和视觉报告，做出最终裁决。
    它接收的已经是干净的Python对象。
    """
    
    cprint("👨‍⚕️ [Phase 3: Final Decision Making] 👨‍⚕️", 'cyan', attrs=['bold'])
    
    # 准备给首席整合官的输入，现在包含三份报告
    synthesis_input = f"""
    [Feature Evolution Report from Cross-Phase Analyst]:
    {text_report}
    [Visual Adjudicator Report from Direct Image Analysis]:
    {visual_report}
    """
    cprint("\n[6. Chief Synthesizer] Synthesizing all two expert reports...", 'magenta')
    final_hybrid_output = agent_6.chat(prompt_text=synthesis_input)

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
    
    return prose_report, structured_summary


# 负责定义agent以及agent的并行调度（每个agent都是独立处理单个切片）
def run_phase_4(text_reports_by_slice, visual_reports_by_slice, selected_slice_numbers):
    """
    【经理函数】执行第三阶段：并行地为每个切片运行首席整合官。
    """
    cprint("\n" + "="*60, 'cyan')
    cprint("👨‍⚕️ [Phase 3: Final Per-Slice Adjudication (Parallel)] 👨‍⚕️", 'cyan', attrs=['bold'])
    cprint("="*60, 'cyan')

    # ... (Agent初始化部分不变) ...
    # Agent 5: 首席整合官
    chief_synthesizer_prompt = f"""
You are the **Chief Radiologist** presiding over an AI diagnostic committee. Your task is to provide the final, global conclusion by synthesizing two expert reports.

**Your Inputs:**
1.  A **Feature Evolution Report** from the Cross-Phase Analyst (summarizing text-based findings over time).
2.  A **Visual Adjudicator Report** (based on direct analysis of all images, with confidence scores).

**Your Reasoning Process:**
Your primary job is to synthesize two expert perspectives to form a final, reasoned conclusion for a single CT slice. The two reports—one from the text-based analysis and one from the direct visual analysis—should be considered **equally important sources of evidence**. You must follow this **Balanced Adjudication Protocol**:

1.  **Identify Discrepancies:** For each of the features, first identify if there is a disagreement between the conclusion from the text-based analysis and the direct visual analysis.

2.  **Handle Agreement:** If the two reports agree, the finding is confirmed. Your justification should reflect this strong consensus.

3.  **Handle Conflict (Balanced Weighing):** If the reports disagree, you must act as an impartial judge. Do not automatically favor one report over the other. Instead, you must weigh the evidence from both sides:
    * **Compare Confidence:** Evaluate the confidence score provided by the Visual Adjudicator. How does it compare to the implied confidence of the text-based report (e.g., was it a strong consensus among the initial analysts)?
    * **Compare Justification:** Critically read the justification and evidence from **both** reports. Which one provides more specific, compelling, and clinically sound reasoning? Does the visual report cite clear, unambiguous image features? Does the text-based report point to a consistent finding across multiple initial analyses?

4.  **Make a Weighed Decision & Justify:** Based on your evaluation, your final answer should reflect the finding with the **stronger overall evidence**, considering both confidence and the quality of the justification. In your final `justification` for that feature, you MUST explain how you resolved the conflict.
    * If the reports agreed, state the consensus. (e.g., *"Confirmed by both text-based analysis and direct visual review."*)
    * If there was a conflict that you resolved, explain your decision.

Your final output MUST be a single block of text that strictly follows the three-part structure outlined below. It is CRITICAL that you include the exact headings for each section, including the numbering.

**Required Output Format:**
1.  **Core Conclusion:** 
    A concise paragraph summarizing the overall clinical findings and conclusion.

2.  **Main Evidence:**
    A bulleted points detailing the key evidence from both the longitudinal and visual reports that support your core conclusion.

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
        "justification": "The text-based analysis from cross-phase reports consistently identified a thick, enhancing capsule in the portal venous phase. This consensus is upheld over a conflicting visual report that did not meet the criteria for an override."
    }},
    "pattern_2": {{
        "answer": 0,
        "justification": "Overriding the inferential text-based finding. The Visual Adjudicator reported with maximum confidence (5/5) that this feature is non-assessable due to the surrounding liver parenchyma not being visible in the images."
    }},
    "pattern_7": {{
        "answer": 0,
        "justification": "Consensus across all reports. The lesion's internal enhancement was described as chaotic and heterogeneous, lacking a nodule-in-nodule appearance."
    }}
}}
"""

    agent_6 = Agent(instruction=chief_synthesizer_prompt, role="Chief Synthesizer")
    final_adjudicated_reports = {}

    final_prose_reports = {}
    final_structured_summaries = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_slice = {}

        for slice_num in selected_slice_numbers:
            # 直接从字典中获取已经处理好的报告
            text_report = text_reports_by_slice.get(slice_num)
            visual_report = visual_reports_by_slice.get(slice_num)

            if not text_report or not visual_report or "error" in str(text_report) or "error" in str(visual_report):
                continue

            # 提交并行任务
            future = executor.submit(
                _adjudicate_single_slice, 
                agent_6, 
                slice_num, 
                text_report, 
                visual_report
            )
            future_to_slice[future] = slice_num
        
        # 5. 收集结果
        cprint(f"Submitting {len(future_to_slice)} final adjudication tasks to run in parallel...", 'magenta')
        for future in tqdm(as_completed(future_to_slice), total=len(future_to_slice), desc="Finalizing Slices"):
            slice_num = future_to_slice[future]
            prose_report, structured_summary = future.result()
            final_prose_reports[slice_num] = prose_report
            final_structured_summaries[slice_num] = structured_summary
    
    final_structured_summaries = dict(sorted(final_structured_summaries.items(), key=lambda item: int(item[0])))

    print(f"final output:")
    print(json.dumps(final_structured_summaries, indent=2, ensure_ascii=False))
    # 函数现在返回一个包含两个字典的元组
    return final_prose_reports, final_structured_summaries

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————




# --- 5. 主执行逻辑 (Main Execution Logic) ---
def main():
    # (这是主函数，负责编排整个流程)
    all_patient_results = {}
    all_selected_paths = {}

    result_filename = "fine_grained_results.json"
    selected_slice_filename = "selected_image_paths.json"
    
    # 在开始处理前，通过写入一个空的JSON对象来清空/初始化文件
    try:
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4)
        cprint(f"✅ Initialized/cleared the results file: '{result_filename}'", 'cyan')
    except Exception as e:
        cprint(f"❌ Failed to initialize results file. Error: {e}", 'red')
        return

    try:
        with open(selected_slice_filename, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4)
        cprint(f"✅ Initialized/cleared the results file: '{selected_slice_filename}'", 'cyan')
    except Exception as e:
        cprint(f"❌ Failed to initialize selected_slice_filename. Error: {e}", 'red')
        return
        
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
        selected_numbers = [int(num) for num in selected_numbers]


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
        # phase1_report = run_phase_1(selected_ap, selected_dp, selected_pvp)
        
        # 执行第二阶段
        # long_report= run_phase_2(phase1_report, selected_numbers)

        # 执行第三阶段
        visual_report = run_phase_3(selected_ap, selected_dp, selected_pvp, selected_numbers)

        # 执行第四阶段
        # prose_report, structured_summary = run_phase_4(long_report, visual_report, selected_numbers)
        try:
            # 1. 先读取文件中的现有数据
            with open(result_filename, 'r', encoding='utf-8') as f:
                all_patient_results = json.load(f)
            # 收集结果
            # all_patient_results[patient_id] = structured_summary
            all_patient_results[patient_id] = visual_report

            # 3. 将更新后的完整数据写回文件
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(all_patient_results, f, ensure_ascii=False, indent=4)

            cprint(f"✅ Saved results for patient {patient_id} to '{result_filename}'", 'green')
        
        except Exception as e:
            cprint(f"❌ Failed to save incremental result for patient {patient_id}. Error: {e}", 'red')

        try:
            with open(selected_slice_filename, 'w', encoding='utf-8') as f:
                json.dump(all_selected_paths, f, indent=4)
            cprint(f"✅ Initialized/cleared paths file: '{selected_slice_filename}'", 'cyan')
        except Exception as e:
            cprint(f"❌ Failed to initialize {selected_slice_filename}. Error: {e}", 'red')
            return

    
if __name__ == '__main__':
    main()
