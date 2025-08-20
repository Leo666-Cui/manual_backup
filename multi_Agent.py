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

# --- 临床问题定义 (Clinical Question Definitions) ---
# 将问题列表定义为全局常量，供所有Agent访问
FEATURE_DEFINITIONS = [
    {
        "name": "Enhancing Capsule",
        "options": [
            'A distinct, hyper-enhancing rim is NOT identified in the PVP/DP, or any visible rim does not show clear enhancement compared to the AP.',
            'By comparing phases, a smooth rim is identified that enhances to become distinctly hyper-enhancing in the PVP or DP.'
        ]
    },
    {
        "name": "Peritumoral Perfusion Alteration",
        "options": [
            'No clear perfusion anomalies are seen in the AP, or any observed hyperenhancement around the lesion persists into the PVP.',
            'A transient perfusion anomaly is confirmed: wedge-shaped or halo-like hyperenhancement is visible around the lesion in the AP and resolves (disappears) in the PVP.'
        ]
    },
    {
        "name": "Corona Enhancement",
        "options": [
            'No radiating vascular pattern is seen at the tumor periphery in any phase.',
            'A dynamic "corona enhancement" is identified: a radiating vascular pattern appears at the tumor periphery in the late AP or PVP and fades in later phases.'
        ]
    },
    {
        "name": "Fade Enhancement Pattern",
        "options": [
            'Comparing across all phases, the lesion demonstrates a "washout" pattern, becoming hypodense in the PVP or DP.',
            'Comparing across all phases, the lesion demonstrates a "fade" pattern, with its enhancement in the delayed phase remaining similar to or greater than its enhancement in the AP/PVP.'
        ]
    },
    {
        "name": "Nodule-in-Nodule Architecture",
        "options": [
            'Across all phases, the lesion\'s internal enhancement is either homogeneous or chaotically heterogeneous, lacking a clear, stable hierarchical structure.',
            'A "nodule-in-nodule" architecture is confirmed across phases: a smaller nodule shows more intense AP enhancement than the larger parent lesion, and this distinction often persists in later phases.'
        ]
    },
    {
        "name": "Peripheral Washout",
        "options": [
            'After initial AP enhancement, the lesion either does not show washout or shows a non-peripheral (diffuse) washout pattern in the PVP/DP.',
            'After initial AP enhancement, the lesion shows a distinct "peripheral washout" pattern, with only its rim becoming hypoenhancing in the PVP/DP.'
        ]
    },
    {
        "name": "Delayed Central Enhancement",
        "options": [
            'Comparing phases, the central part of the lesion does not show progressive enhancement (e.g., it washes out or remains persistently non-enhancing).',
            'Comparing phases, the central part of the lesion shows progressive, sustained enhancement, becoming brighter in the delayed phase than it was in the AP/PVP.'
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
        
        cprint(f"Agent '{self.role}' is sending request to {self.config['model_name']} with {valid_image_count} image(s)...", 'blue')

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

def run_phase_1(image_paths_ap, image_paths_dp, image_paths_pvp):
    cprint("\n--- [Phase 1: Parallel Feature Extraction with Professional Questions] ---", 'yellow', attrs=['bold'])

    # 动态构建高级Prompt (现在从全局常量 FEATURE_DEFINITIONS 读取)
    prompt_sections = []
    json_findings_template = []
    for i, feature in enumerate(FEATURE_DEFINITIONS):
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
You are a meticulous radiologist. Your task is to analyze a set of CT images from a specific phase.
For each of the {len(FEATURE_DEFINITIONS)} features below, you are presented with two descriptive statements: Option 0 and Option 1.
Your tasks are:
1. Carefully analyze the provided images.
2. For each feature, determine which option (0 or 1) most accurately describes the lesion.
3. Format your final output as a SINGLE, VALID JSON object. Do not add any explanatory text or markdown formatting outside of the JSON structure.
{''.join(prompt_sections)}
Your JSON output must follow this exact structure:
{{
  "phase": "PHASE_ID",
  "findings": [
    {', '.join(json_findings_template)}
  ]
}}
"""
    
    # 3. Agent初始化和调用 (与之前版本相同)
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
    for phase, paths in [("AP", image_paths_ap), ("DP", image_paths_dp), ("PVP", image_paths_pvp)]:
        cprint(f"Running analysis for phase {phase} with professional questions...", 'magenta')
        prompt = questions_prompt_template.replace("PHASE_ID", phase)
        # 根据当前循环选择对应的Agent
        if phase == 'AP':
            agent = agent_1
        elif phase == 'DP':
            agent = agent_2
        else:
            agent = agent_3
        
        response_text = agent.chat(prompt_text=prompt, image_paths=paths)
        
        try:
            # 清理并解析JSON
            clean_json_text = response_text.strip().replace("```json", "").replace("```", "")
            reports[phase] = json.loads(clean_json_text)
            cprint(f"Successfully received and parsed JSON for Phase {phase}.", 'green')
        except (json.JSONDecodeError, AttributeError):
            cprint(f"Error: Failed to parse JSON from Phase {phase} Agent. Response:\n{response_text}", 'red')
            reports[phase] = {"error": "Failed to get a valid JSON response.", "raw_response": response_text}
        
        # print(f"phase1 output: {reports}")
            
    return reports

def run_phase_2(reports_from_phase1, all_image_paths):
    # (此函数负责执行第二和第三阶段的文本分析)
    cprint("\n" + "="*60, 'cyan')
    cprint("🚀 Executing Phases 2 & 3 of the Workflow 🚀", 'cyan', attrs=['bold'])
    
    # 检查第一阶段是否有错误
    if any("error" in r for r in reports_from_phase1.values()):
        cprint("Aborting Phases 2 and 3 due to errors in Phase 1.", 'red')
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

    # Agent 4a: 纵向分析师
    longitudinal_analyst_prompt = """
    You are a senior radiologist acting as a **longitudinal analysis specialist** within an AI diagnostic committee.
    Your input consists of three structured reports generated by phase-specific AI analysts (AP, DP, PVP).
    Your sole task is to analyze these three reports, focusing on the time dimension to summarize the evolution of each feature. Your output will be a concise, feature-centric evolution report **for the final Chief Radiologist**.
    It is important to highlight any inconsistencies in the findings between the different phase reports if they exist.
    **IMPORTANT: Your output must be a direct, Feature-Centric Evolution Report. Do not include any headers, titles, salutations, or conversational text like 'To:', 'From:', or 'Subject:'.**
    """
    agent_4a = Agent(instruction=longitudinal_analyst_prompt, role="Longitudinal Analyst")
    longitudinal_report = agent_4a.chat(prompt_text=committee_input)
    cprint("--- Longitudinal Analyst Output ---", 'green')
    # print(longitudinal_report)

    # Agent 4b: 横向分析师
    cross_sectional_analyst_prompt = """
    You are a senior radiologist acting as a **pattern recognition specialist** within an AI diagnostic committee.
    Your input consists of three structured reports generated by phase-specific AI analysts (AP, DP, PVP).
    Your sole task is to analyze these three reports, focusing on each phase independently to provide a 'diagnostic snapshot'.
    Evaluate the combination of features present to determine the clinical picture for each phase. Your output will be a concise, timepoint-centric snapshot report **for the final Chief Radiologist**.
    """    
    agent_4b = Agent(instruction=cross_sectional_analyst_prompt, role="Cross-sectional Analyst")
    cross_sectional_report = agent_4b.chat(prompt_text=committee_input)
    cprint("--- Cross-sectional Analyst Output ---", 'green')
    # print(cross_sectional_report)


    # Agent 5: 视觉仲裁官
    cprint("\n[5. Visual Adjudicator] Performing direct comparative analysis of all images...", 'magenta')
    # 为Agent 5动态构建Prompt
    prompt_sections = []
    for i, feature in enumerate(FEATURE_DEFINITIONS):
        prompt_sections.append(f"""
--- Feature {i+1}: {feature['name']} ---
Option 0 : "{feature['options'][0]}"
Option 1 : "{feature['options'][1]}"
""")

    visual_adjudicator_task_prompt = f"""
As the Visual Adjudicator, you have access to ALL images from ALL phases (AP, DP, PVP).
Your task is to perform a direct, holistic, comparative analysis of all images to choose the most accurate description for each of the {len(FEATURE_DEFINITIONS)} features below.
Your final output should be a single report in a clear, point-by-point format. For each feature, state your conclusion (Option 0 or 1) and provide a justification based on your direct, multi-phase visual evidence.
{''.join(prompt_sections)}
"""
print(f"visual_adjudicator_task_prompt: \n{visual_adjudicator_task_prompt}")

    visual_adjudicator_instruction = """
    You are an expert radiologist, the **Visual Adjudicator** for an AI diagnostic committee.
    You have been given access to the **complete set of CT images from all phases (AP, DP, PVP)**.
    Your sole task is to perform a direct, comparative analysis of all images to answer the 7 key questions. Base your answers on the holistic visual evidence.
    Your output should be a structured report, detailing your findings for each feature with a clear justification based on your direct observation.

    **Example Output Format:**
    - **Enhancing Capsule:** [Your direct visual finding, e.g., "A clear, enhancing capsule becomes visible in the PVP and DP phases."]
    - **Nodule-in-Nodule Architecture:** [Your direct visual finding, e.g., "No definitive inner nodule with separate enhancement characteristics is identified across any phase."]
    """
    agent_5 = Agent(instruction=visual_adjudicator_instruction, role="Visual Adjudicator")
    # 视觉仲裁官的prompt比较简单，因为它主要依赖于图像输入
    adjudicator_task_prompt = "Please analyze the provided multi-phase images and provide your direct visual findings for the 7 key radiological features."
    visual_adjudicator_report = agent_5.chat(prompt_text=visual_adjudicator_task_prompt, image_paths=all_image_paths)
    cprint("--- Visual Adjudicator (5) Output ---", 'green')
    print(visual_adjudicator_report)

    return longitudinal_report, cross_sectional_report, visual_adjudicator_report



def run_phase_3(longitudinal_report, cross_sectional_report, visual_adjudicator_report):
    """
    执行第三阶段：运行首席整合官，综合所有分析报告。
    """
    cprint("\n--- [Phase 3: Final Decision Making] ---", 'yellow', attrs=['bold'])
    
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
You are the **Chief Radiologist** presiding over an AI diagnostic committee. Your task is to provide the final, global conclusion.
Your input consists of two expert summaries, which were generated by specialist AI analysts:
1.  A **Feature Evolution Report** from the Longitudinal Analyst.
2.  A **Diagnostic Snapshot Report** from the Cross-sectional Analyst.
3.  A **Visual Adjudicator Report** (based on direct analysis of all images).
Your job is to **synthesize these three distinct perspectives** (the 'what changed over time' and the 'what is the pattern now') to form a single, coherent, final report.

Your final output MUST be a single block of text that strictly follows the three-part structure outlined below. It is CRITICAL that you include the exact headings for each section, including the numbering.
**Crucially, if there is a conflict between the text-based analysis (reports 1 & 2) and the direct visual analysis (report 3), you MUST give precedence to the Visual Adjudicator's report as it is based on the primary image evidence.**

1.  **Core Conclusion:** 
    A concise paragraph summarizing the overall clinical findings and conclusion.

2.  **Main Evidence:**
    A bulleted points detailing the key evidence from both the longitudinal and cross-sectional reports that support your core conclusion.

3.  **Structured Summary:**
    This section MUST be a single, valid JSON object and nothing else. Do not add any introductory text, markdown tags like ```json, or any text after the JSON object.
    The JSON object must summarize the final answer for each of the 7 features. It must have keys from "pattern_1" to "pattern_7", corresponding to the features in this order:
    1. Enhancing Capsule
    2. Peritumoral Perfusion Alteration
    3. Corona Enhancement
    4. Fade Enhancement Pattern
    5. Nodule-in-Nodule Architecture
    6. Peripheral Washout
    7. Delayed Central Enhancement

    For each pattern, the value must be an object with two keys:
    - "answer": The final binary conclusion (0 for first description, 1 for second description).
    - "justification": A brief, concise summary of the reasoning.

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
    "pattern_5": {{
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
    # print(structured_summary)

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

    for patient_id in tqdm(patient_ids, desc="Processing Patients"):
        cprint(f"\n{'='*20} Processing Patient: {patient_id} {'='*20}", 'yellow')
        patient_folder = os.path.join(BASE_DATA_PATH, patient_id)
        
        # 为每个时期收集图像路径
        image_paths_ap = sorted([os.path.join(patient_folder, 'AP', f) for f in os.listdir(os.path.join(patient_folder, 'AP')) if f.endswith('.png')])
        image_paths_dp = sorted([os.path.join(patient_folder, 'DP', f) for f in os.listdir(os.path.join(patient_folder, 'DP')) if f.endswith('.png')])
        image_paths_pvp = sorted([os.path.join(patient_folder, 'PVP', f) for f in os.listdir(os.path.join(patient_folder, 'PVP')) if f.endswith('.png')])
        
        if not all([image_paths_ap, image_paths_dp, image_paths_pvp]):
            cprint(f"Warning: Patient {patient_id} is missing images in one or more phase folders. Skipping.", 'red')
            continue

        # 执行第一阶段
        phase1_reports = run_phase_1(image_paths_ap, image_paths_dp, image_paths_pvp)
        # print(f"phase1 reports: {phase1_reports}")
        
        # 2. 执行第二阶段
        all_images = image_paths_ap + image_paths_dp + image_paths_pvp
        long_report, cross_report, visual_report = run_phase_2(phase1_reports, all_images)
        
        # 3. 执行第三阶段
        prose_report, structured_summary = run_phase_3(long_report, cross_report, visual_report)
        
        # 收集结果
        all_patient_results[patient_id] = structured_summary

        # cprint(f"\n--- Final Hybrid Report for Patient {patient_id} ---", 'cyan', attrs=['bold'])
        # print("--- Prose Report ---")
        # print(prose_report)
        # print("\n--- Structured Summary (JSON) ---")
        # print(json.dumps(structured_summary, indent=4, ensure_ascii=False))

    # 将所有病人的结果保存到一个JSON文件中
    output_filename = "multi_agent_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_patient_results, f, ensure_ascii=False, indent=4)
    
    cprint(f"\n\n🎉 All processing complete. All reports have been saved to '{output_filename}'.", 'cyan', attrs=['bold'])


if __name__ == '__main__':
    main()
