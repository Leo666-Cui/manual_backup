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

def find_core_slice(image_paths_pvp: list):
    """
    使用两阶段AI流程，从一个期相的图片列表中智能选出“核心切片”。
    
    :param image_paths_pvp: PVP期相的10张图片路径列表。
    :return: 选出的核心切片号码 (int)，如果失败则返回 None。
    """
    cprint("\n--- [启动核心切片智能筛选子流程] ---", 'blue', attrs=['bold'])

    # --- 初始化本流程所需的Agent ---
    
    # 1a. 切片评分员 (Slice Scorer)
    slice_scorer_instruction = """
    You are a radiologist assistant. Your task is to evaluate a single CT slice and provide scores for its diagnostic quality.
    Based on the single image provided, please rate the following three criteria on a scale of 1 (very poor) to 10 (excellent).
    Your output must be a single, valid JSON object with no other text.
    1.  Lesion Size Score: How large is the lesion in this slice?
    2.  Border Clarity Score: How clear and well-defined is the lesion's border?
    3.  Feature Conspicuousness Score: How clearly are features visible?
    """
    slice_scorer_agent = Agent(instruction=slice_scorer_instruction, role="Slice Scorer")

    # 1b. 甄选决策官 (Selection Judge)
    selection_judge_instruction = """
    You are a senior radiologist. You have been provided with scorecards for multiple CT slices.
    Your task is to select the single best 'Core Slice' by weighing all scores, with a slight priority for 'Border Clarity' and 'Feature Conspicuousness'.
    Your final output must be a single number representing the chosen slice number, inside a JSON object.
    """
    selection_judge_agent = Agent(instruction=selection_judge_instruction, role="Selection Judge")

    # --- 阶段一：并行评分 (“海选”) ---
    scorecards = []
    cprint("--- [海选阶段] 开始对PVP切片进行评分...", 'blue')
    # 在实际应用中，这里的循环可以使用 threading 或 asyncio 实现真正的并行处理
    for image_path in tqdm(image_paths_pvp, desc="Scoring PVP Slices"):
        
        # 从文件名中提取切片号
        slice_number = None
        match = re.search(r'(\d+)\.png$', image_path)
        if match:
            slice_number = int(match.group(1))
        
        if slice_number is None:
            cprint(f"Warning: Could not extract slice number from '{image_path}'. Skipping.", 'yellow')
            continue

        # 为评分员构建专属任务Prompt
        scorer_task_prompt = f"""
        Please evaluate the provided CT slice and output your scores in the required JSON format.
        This is slice number {slice_number}.

        REQUIRED JSON OUTPUT FORMAT:
        {{
            "slice_number": {slice_number},
            "size_score": "[Your score from 1-10]",
            "border_score": "[Your score from 1-10]",
            "feature_score": "[Your score from 1-10]"
        }}
        """
        
        # 调用评分员Agent，注意image_paths是一个只包含单张图片的列表
        response_text = slice_scorer_agent.chat(prompt_text=scorer_task_prompt, image_paths=[image_path])
        
        # 解析评分结果
        try:
            clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
            scorecard = json.loads(clean_json_text)
            scorecards.append(scorecard)
        except (json.JSONDecodeError, AttributeError):
            cprint(f"Error: Failed to parse JSON from Slice Scorer for '{image_path}'. Response:\n{response_text}", 'red')
    # print(f"scorecards: \n{scorecards}\n")

    # --- 阶段二：综合决策 (“决选”) ---
    if not scorecards:
        cprint("Error: No valid scorecards were generated in the scoring phase. Cannot determine core slice.", 'red')
        return None

    cprint("\n--- [决选阶段] 开始根据评分选择核心切片...", 'blue')
    
    # 为决策官准备输入
    judge_task_prompt = f"""
    Here are the scorecards for {len(scorecards)} different CT slices. Please select the single best 'Core Slice' number.

    **Scorecards:**
    {json.dumps(scorecards, indent=2)}

    **REQUIRED JSON OUTPUT FORMAT:**
    {{
        "core_slice_number": [The number of the slice you selected]
    }}
    """
    
    # 调用决策官Agent
    response_text = selection_judge_agent.chat(prompt_text=judge_task_prompt)
    
    # 解析最终决策
    try:
        clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(clean_json_text)
        core_slice_number = int(decision["core_slice_number"])
        cprint(f"✅ AI has selected Core Slice Number: {core_slice_number}", 'green', attrs=['bold'])
        return core_slice_number
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        cprint(f"Error: Failed to parse JSON from Selection Judge. Error: {e}. Response:\n{response_text}", 'red')
        return None


def select_final_slices(core_slice_num, image_paths_ap, image_paths_dp, image_paths_pvp):
    """
    根据AI选出的核心切片号码，智能地从每个期相中选择三张代表性切片。
    - 如果核心切片是第一张，则选择前三张。
    - 如果核心切片是最后一张，则选择最后三张。
    - 否则，选择核心切片及其前后两张。

    :param core_slice_num: AI选出的核心切片号码 (int)。
    :param image_paths_ap, dp, pvp: 三个期相的完整图片路径列表。
    :return: 一个包含三个列表的元组 (selected_ap, selected_dp, selected_pvp)。
    """
    # cprint(f"Core slice {core_slice_num} selected. Selecting adjacent slices based on file order...", 'cyan')

    # --- 步骤 1: 在PVP列表中找到核心切片的索引位置 ---
    # 我们以PVP作为定位基准
    core_slice_index = -1
    
    # 首先，解析所有PVP路径以获取号码和索引的映射
    pvp_slice_numbers = []
    for path in image_paths_pvp:
        match = re.search(r'(\d+)\.png$', path)
        if match:
            pvp_slice_numbers.append(int(match.group(1)))
        else:
            pvp_slice_numbers.append(-1) # 添加一个无效值以保持索引对应

    try:
        # 找到核心号码在号码列表中的第一个匹配项的索引
        core_slice_index = pvp_slice_numbers.index(core_slice_num)
    except ValueError:
        cprint(f"Error: Could not find the core slice number {core_slice_num} in the PVP image list. Skipping.", 'red')
        return None, None, None

    # --- 步骤 2: 根据您的新规则，确定最终的切片窗口索引 ---
    num_images = len(image_paths_pvp)
    
    if core_slice_index == 0:
        # 如果核心切片是第一张，选择前三张
        start_index = 0
        end_index = 3
        cprint("Core slice is the first slice. Selecting the first three.", "blue")
    elif core_slice_index == num_images - 1:
        # 如果核心切片是最后一张，选择最后三张
        start_index = num_images - 3
        end_index = num_images
        cprint("Core slice is the last slice. Selecting the last three.", "blue")
    else:
        # 否则，选择核心切片及其邻居
        start_index = core_slice_index - 1
        end_index = core_slice_index + 2
        cprint("Core slice is in the middle. Selecting adjacent three.", "blue")
    
    # 再次确保索引不会因列表太短而出错
    start_index = max(0, start_index)
    end_index = min(num_images, end_index)


    # --- 步骤 3: 将这个安全的索引范围，统一应用到所有三个期相的列表上 ---
    selected_paths_ap = image_paths_ap[start_index:end_index]
    selected_paths_dp = image_paths_dp[start_index:end_index]
    selected_paths_pvp = image_paths_pvp[start_index:end_index]

    cprint(f"Selected AP slices: {[os.path.basename(p) for p in selected_paths_ap]}", 'cyan')
    cprint(f"Selected DP slices: {[os.path.basename(p) for p in selected_paths_dp]}", 'cyan')
    cprint(f"Selected PVP slices: {[os.path.basename(p) for p in selected_paths_pvp]}", 'cyan')

    return selected_paths_ap, selected_paths_dp, selected_paths_pvp

def run_phase_1(image_paths_ap, image_paths_dp, image_paths_pvp):
    cprint("\n--- [Phase 1: Parallel Feature Extraction with Professional Questions] ---", 'yellow', attrs=['bold'])
    
    # 3. Agent初始化和调用 
    # 为动脉期(AP)分析师创建指令
    ap_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze a set of CT images from the **Arterial Phase (AP)**.
    Your output will be a structured JSON report. **It is CRITICAL that your JSON is perfectly formatted and self-contained because it will be the direct input for other specialist AI analysts who will perform longitudinal and cross-sectional analysis. Their success depends entirely on the accuracy and clarity of your report.**
    You must answer all the specific questions provided below and format your output as a JSON object.
    Do not add any explanatory text outside of the JSON structure.
    For the 'value' key, use 0 for the first description or 1 for the second description.
    Provide a justification for your finding in the 'evidence' key.
    """

    # 为延迟期(DP)分析师创建指令
    dp_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze a set of CT images from the **Delayed Phase (DP)**.
    Your output will be a structured JSON report. **It is CRITICAL that your JSON is perfectly formatted and self-contained because it will be the direct input for other specialist AI analysts who will perform longitudinal and cross-sectional analysis. Their success depends entirely on the accuracy and clarity of your report.**
    You must answer all the specific questions provided below and format your output as a JSON object.
    Do not add any explanatory text outside of the JSON structure.
    For the 'value' key, use 0 for the first description or 1 for the second description.
    Provide a justification for your finding in the 'evidence' key.
    """

    # 为门脉期(PVP)分析师创建指令
    pvp_analyst_instruction = """
    You are a meticulous radiologist. Your task is to analyze a set of CT images from the **Portal Venous Phase (PVP)**.
    Your output will be a structured JSON report. **It is CRITICAL that your JSON is perfectly formatted and self-contained because it will be the direct input for other specialist AI analysts who will perform longitudinal and cross-sectional analysis. Their success depends entirely on the accuracy and clarity of your report.**
    You must answer all the specific questions provided below and format your output as a JSON object.
    Do not add any explanatory text outside of the JSON structure.
    For the 'value' key, use 0 for the first description or 1 for the second description.
    Provide a justification for your finding in the 'evidence' key.
    """

    agent_1 = Agent(instruction=ap_analyst_instruction, role="phase AP Analyst")
    agent_2 = Agent(instruction=dp_analyst_instruction, role="phase DP Analyst")
    agent_3 = Agent(instruction=pvp_analyst_instruction, role="phase PVP Analyst")

    reports = {}
    for phase, paths, agent in [("AP", image_paths_ap, agent_1), ("DP", image_paths_dp, agent_2), ("PVP", image_paths_pvp, agent_3)]:
        
        # 1. 【【核心逻辑】】为当前期相筛选相关问题
        questions_for_this_phase = []
        for feature in FEATURE_DEFINITIONS:
            # 查询知识库，看当前特征是否需要本期相
            if phase in FEATURE_PHASE_REQUIREMENTS.get(feature['name'], []):
                questions_for_this_phase.append(feature)

        if not questions_for_this_phase:
            cprint(f"No relevant questions for phase {phase}. Skipping agent call.", 'yellow')
            # 为该期相创建一个空的报告
            reports[phase] = {"phase": phase, "findings": []}
            continue

        cprint(f"Running analysis for phase {phase}. Asking {len(questions_for_this_phase)} relevant question(s)...", 'magenta')

        # 2. 动态构建只包含相关问题的高级Prompt
        prompt_sections = []
        json_findings_template = []
        for i, feature in enumerate(questions_for_this_phase):
            prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Option 0 : "{feature['options'][0]}"
Option 1 : "{feature['options'][1]}"
""")
            json_findings_template.append(
                f'{{"feature": "{feature["name"]}", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}}'
            )

        # 构建完整的、新的Prompt模板
        questions_prompt_template = f"""
You are a meticulous radiologist analyzing images from the {phase} phase.
For each of the {len(questions_for_this_phase)} features below, you are presented with two descriptive statements.
Your task is to analyze the provided images and determine which option (0 or 1) is most accurate for each feature.
Format your final output as a SINGLE, VALID JSON object.
{''.join(prompt_sections)}
Your JSON output must follow this exact structure:
{{
  "phase": "{phase}",
  "findings": [
    {', '.join(json_findings_template)}
  ]
}}
"""
        response_text = agent.chat(prompt_text=questions_prompt_template, image_paths=paths)
        try:
            # 清理并解析JSON
            clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
            reports[phase] = json.loads(clean_json_text)
            cprint(f"Successfully received and parsed JSON for Phase {phase}.", 'green')
        except (json.JSONDecodeError, AttributeError):
            cprint(f"Error: Failed to parse JSON from Phase {phase} Agent. Response:\n{response_text}", 'red')
            reports[phase] = {"error": "Failed to get a valid JSON response.", "raw_response": response_text}
        
    # print(f"phase_1 output: {reports}")
    return reports


def run_phase_2(reports_from_phase1):
    # (此函数负责执行第二和第三阶段的文本分析)
    cprint("\n" + "="*60, 'cyan')
    cprint("🚀 Executing Phases 2 of the Workflow 🚀", 'cyan', attrs=['bold'])
    
    # 检查第一阶段是否有错误
    if any("error" in r for r in reports_from_phase1.values()):
        cprint("Aborting Phases 2 due to errors in Phase 1.", 'red')
        return {"error": "Phase 1 failed.", "details": reports_from_phase1}
        
    cprint("="*60, 'cyan')
    cprint("\n--- [Phase 2: The Senior Analysis Committee] ---", 'yellow', attrs=['bold'])

    committee_input = f"""
    Below are three structured reports from phase Analysts regarding CT images from three different periods:
    [Report from phase AP]:
    {json.dumps(reports_from_phase1['AP'], indent=2, ensure_ascii=False)}
    [Report from phase DP]:
    {json.dumps(reports_from_phase1['DP'], indent=2, ensure_ascii=False)}
    [Report from phase PVP]:
    {json.dumps(reports_from_phase1['PVP'], indent=2, ensure_ascii=False)}
    """

    # Agent 4a: 纵向分析师：每个问题在三个时期的变化
    longitudinal_analyst_prompt = """
    You are a senior radiologist acting as a **cross-phase analysis specialist** within an AI diagnostic committee.
    Your input consists of three **partial, phase-specific reports** from AI analysts (AP, DP, PVP). A report for a given phase will only contain findings for features observable in that phase. 

    Your task is to **synthesize these partial reports to create a complete evolutionary summary for all 9 features**.
    For any given feature, you may only have input from one, two, or all three reports; this is expected. You must deduce the overall evolution based on the available information. For example, to determine "Peritumoral Perfusion Alteration", you should primarily rely on the AP and PVP reports. 

    **IMPORTANT: Do not include any headers, titles, salutations, or conversational text like 'To:', 'From:', or 'Subject:'.**
    Your output will be a concise, feature-centric evolution report for the final Chief Radiologist. It is important to highlight any inconsistencies in the findings if they exist.
    """
    agent_4a = Agent(instruction=longitudinal_analyst_prompt, role="Longitudinal Analyst")

    # Agent 4b: 横向分析师：每个时期所有问题特征合集
    cross_sectional_analyst_prompt = """
    You are a senior radiologist acting as a **pattern recognition specialist** within an AI diagnostic committee.
    Your input consists of three **partial, phase-specific reports** from AI analysts (AP, DP, PVP). Each report represents a 'snapshot' of findings for that specific phase.

    Your task is to review these individual snapshots and create a summary report. For each phase (AP, DP, PVP), describe the overall clinical picture based on the combination of features reported for that phase. **Highlight the key diagnostic contribution of each phase.**
    For example, you might state that the AP phase was crucial for identifying hyperenhancement, while the PVP and DP phases were definitive for confirming washout and an enhancing capsule.

    Your output will be a concise, phase-centric snapshot report for the final Chief Radiologist.
    """
    agent_4b = Agent(instruction=cross_sectional_analyst_prompt, role="Cross-sectional Analyst")

    # --- Execute Agents in Parallel ---
    longitudinal_report = None
    cross_sectional_report = None

    # agent_4a，agent_4并行运行
    with ThreadPoolExecutor(max_workers=2) as executor:
        cprint("Submitting Agent 4a and 4b tasks to run in parallel...", 'blue')
        
        # Submit each agent's chat method as a separate task
        future_4a = executor.submit(agent_4a.chat, prompt_text=committee_input)
        future_4b = executor.submit(agent_4b.chat, prompt_text=committee_input)
        
        # Retrieve the results as they complete
        # future.result() will wait for the task to finish
        longitudinal_report = future_4a.result()
        cross_sectional_report = future_4b.result()

    cprint("--- Longitudinal Analyst Output ---", 'green')
    # print(longitudinal_report)
    cprint("--- Cross-sectional Analyst Output ---", 'green')
    # print(cross_sectional_report)
    return longitudinal_report, cross_sectional_report


def run_phase_visual(all_image_paths):
    # (此函数负责执行第二和第三阶段的文本分析)
    cprint("\n" + "="*60, 'cyan')
    cprint("👀 Executing Visual Adjudicator 👀", 'cyan', attrs=['bold'])

    # Agent 5: 视觉仲裁官
    cprint("\n[5. Visual Adjudicator] Performing direct comparative analysis of all images...", 'magenta')
    # 为Agent 5动态构建Prompt
    prompt_sections = []
    for i, feature in enumerate(FEATURE_DEFINITIONS):
        prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Answer 0 : "{feature['options'][0]}"
Answer 1 : "{feature['options'][1]}"
""")

    visual_adjudicator_task_prompt = f"""
You are an expert radiologist, the **Visual Adjudicator** for an AI diagnostic committee. Your judgment is final as it is based on direct visual evidence.

**IMAGE CONTEXT:**
You have been provided with a sequence of 9 CT images(3 CT images for each phase). The sequence is ordered by phase:
- **Images 1-3 ** belong to the **Arterial Phase (AP)**.
- **Images 4-6 ** belong to the **Delayed Phase (DP)**.
- **Images 7-9 ** belong to the **Portal Venous Phase (PVP)**.

**CONFIDENCE SCORING RUBRIC (MUST FOLLOW):**
You MUST use the following scale to determine the confidence score. Be very conservative; start with a baseline of 3 and only increase if the evidence is exceptionally strong.
- **5 (Extremely Certain):** The finding is unambiguous, textbook-perfect, and clearly visible on multiple slices or phases. There is no other reasonable interpretation.
- **4 (Very Certain):** The finding is clear and fits the definition, but may not be perfectly "textbook" or has very minor ambiguity.
- **3 (Moderately Certain):** The finding is suggestive but not definitive. Features support this conclusion, but other interpretations are possible. This should be your default for subtle findings.
- **2 (Uncertain):** The evidence is weak, subtle, or could be an imaging artifact. A finding is suspected but cannot be confirmed.
- **1 (Very Uncertain/Guess):** There is virtually no direct evidence.

**YOUR TASK:**
Perform a direct, holistic, and comparative analysis of all images to answer the {len(FEATURE_DEFINITIONS)} key questions. For each feature, you must provide an answer (0 or 1), a confidence score (1-5), and a justification that cites specific image numbers as evidence.
Your final output MUST be a SINGLE, VALID JSON object. 

**FEW-SHOT EXAMPLES:**
```json
{{
    "pattern_1": {{
        "answer": 1,
        "confidence": 5,
        "justification": "For this positive finding, review of the Delayed Phase (DP) sequence (images 4-6) reveals a distinct and smooth hyper-enhancing rim completely encircling the lesion, which was not visible in the Arterial Phase."
    }},
    "pattern_2": {{
        "answer": 0,
        "confidence": 4,
        "justification": "For this negative finding, review of the Arterial Phase (AP) sequence (images 1-3) shows homogeneous enhancement in the liver parenchyma surrounding the lesion, with no evidence of the characteristic wedge-shaped or halo-like hyperenhancement."
    }}
}}

**QUESTIONS TO ANSWER:**
{''.join(prompt_sections)}
"""

    visual_adjudicator_instruction = "You are an expert radiologist, the **Visual Adjudicator** for an AI diagnostic committee. Follow all instructions in the user prompt precisely."
    agent_5 = Agent(instruction=visual_adjudicator_instruction, role="Visual Adjudicator")
    # 视觉仲裁官的prompt比较简单，因为它主要依赖于图像输入
    visual_adjudicator_report = agent_5.chat(prompt_text=visual_adjudicator_task_prompt, image_paths=all_image_paths)
    cprint("--- Visual Adjudicator (5) Output ---", 'green')
    # print(visual_adjudicator_report)

    return visual_adjudicator_report


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

# --- 5. 主执行逻辑 (Main Execution Logic) ---

def main():
    # (这是主函数，负责编排整个流程)
    all_patient_results = {}

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


        # 2. 调用AI来寻找核心切片
        core_slice_num = find_core_slice(image_paths_pvp)

        if core_slice_num is None:
            cprint(f"Could not determine a core slice for patient {patient_id}. Skipping.", 'red')
            all_patient_results[patient_id] = {"error": "Core slice selection failed."}
            continue
            
        # 3. 选出visual agent每个时期应该看的3张图
        selected_paths_ap, selected_paths_dp, selected_paths_pvp = select_final_slices(
            core_slice_num, image_paths_ap, image_paths_dp, image_paths_pvp
        )

        # 如果选择失败（例如核心切片未在列表中找到），则跳过
        if selected_paths_ap is None:
            all_patient_results[patient_id] = {"error": f"Core slice {core_slice_num} could not be located in the image lists."}
            continue


        # 执行第一阶段，每时期10张图
        phase1_reports = run_phase_1(image_paths_ap, image_paths_dp, image_paths_pvp)
        # print(f"phase1 reports: {phase1_reports}")
        
        # 2. 执行第二阶段
        all_images = selected_paths_ap + selected_paths_dp + selected_paths_pvp
        long_report, cross_report = run_phase_2(phase1_reports)
        # 每时期3张图
        visual_report = run_phase_visual(all_images)
        print(f"-------------long_report:------------- \n{long_report}\n")
        print(f"-------------cross_report:------------- \n{cross_report}\n")
        print(f"-------------visual_report:------------- \n{visual_report}\n")
        
        # 3. 执行第三阶段
        prose_report, structured_summary = run_phase_3(long_report, cross_report, visual_report)
        
        # 收集结果
        all_patient_results[patient_id] = structured_summary


    # 将所有病人的结果保存到一个JSON文件中
    output_filename = "multi_agent_external_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_patient_results, f, ensure_ascii=False, indent=4)
    
    cprint(f"\n\n🎉 All processing complete. All reports have been saved to '{output_filename}'.", 'cyan', attrs=['bold'])


if __name__ == '__main__':
    main()
