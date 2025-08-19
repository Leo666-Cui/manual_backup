import os
import json
import base64
import requests
from tqdm import tqdm
from termcolor import cprint
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

# --- 1. 配置区 (Configuration Section) ---
# 请在此处修改您的个人配置

# 项目和模型配置
VERTEX_AI_CONFIG = {
    "project_id": "gen-lang-client-0514315738",
    "location": "us-central1",
    # 【【【重要】】】请务必修改为您的服务账号密钥文件的真实路径
    "credentials_path": "/home/yxcui/FM-Bridge/testing_file/gen-lang-client-0514315738-faeaf04b384e.json",
    "model_name": "gemini-2.5-flash" 
}

# 代理配置
PROXY_CONFIG = {
    # 如果您不需要代理，请将此行设置为 None, 例如：'https': None
    'https': 'socks5://hkumedai:bestpaper@66.42.72.8:54321' 
}

# 【【【重要】】】请修改为您的数据集根目录
BASE_DATA_PATH = "/home/yxcui/FM-Bridge/testing_file/test_dataset/cropped_20_slices_image"


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

def run_phase1_analysis(image_paths_ap, image_paths_dp, image_paths_pvp):
    # (此函数负责执行第一阶段的真实VLLM调用)
    cprint("\n--- [Phase 1: Parallel Feature Extraction with Professional Questions] ---", 'yellow', attrs=['bold'])

    # 1. 【【新整合的代码】】定义专业的特征名称和对应的描述选项
    # (这里我们手动提取了您注释中的特征名称)
    feature_names = [
        "Enhancing Capsule",
        "Peritumoral Perfusion Alteration",
        "Corona Enhancement",
        "Fade Enhancement Pattern",
        "Nodule-in-Nodule Architecture",
        "Peripheral Washout",
        "Delayed Central Enhancement"
    ]

    questions_list_professional = [
        # --- 1. 强化的包膜 (Enhancing Capsule) ---
        # 注释: 一个真正的强化包膜，其特征在于它在PVP或DP相对于AP的“强化”行为。因此，必须对比期相来确认。
        ['A distinct, hyper-enhancing rim is NOT identified in the PVP/DP, or any visible rim does not show clear enhancement compared to the AP.',
        'By comparing phases, a smooth rim is identified that enhances to become distinctly hyper-enhancing in the PVP or DP.'],

        # --- 2. 瘤周异常灌注 (Peritumoral Perfusion Alteration) ---
        # 注释: 该特征的本质是“一过性”的，即来得快去得也快。因此，必须通过AP和PVP的对比来确认其是否“一过性”。
        ['No clear perfusion anomalies are seen in the AP, or any observed hyperenhancement around the lesion persists into the PVP.',
        'A transient perfusion anomaly is confirmed: wedge-shaped or halo-like hyperenhancement is visible around the lesion in the AP and resolves (disappears) in the PVP.'],

        # --- 3. 冠状强化 (Corona Enhancement) ---
        # 注释: “冠状强化”描述的是一种动态的血流现象，观察它在不同期相的出现和消退才能最终确认。
        ['No radiating vascular pattern is seen at the tumor periphery in any phase.',
        'A dynamic "corona enhancement" is identified: a radiating vascular pattern appears at the tumor periphery in the late AP or PVP and fades in later phases.'],

        # --- 4. "Fade" 强化模式 ---
        # 注释: “Fade”模式的定义本身就是一个跨越所有期相的比较过程，与“Washout”相对。
        ['Comparing across all phases, the lesion demonstrates a "washout" pattern, becoming hypodense in the PVP or DP.',
        'Comparing across all phases, the lesion demonstrates a "fade" pattern, with its enhancement in the delayed phase remaining similar to or greater than its enhancement in the AP/PVP.'],

        # --- 5. 结中结模式 (Nodule-in-Nodule Architecture) ---
        # 注释: 确认“结中结”不仅要看AP期的形态，还要看母子结节在PVP和DP的不同动态行为（如廓清程度不同），才能做出最可靠的判断。
        ['Across all phases, the lesion\'s internal enhancement is either homogeneous or chaotically heterogeneous, lacking a clear, stable hierarchical structure.',
        'A "nodule-in-nodule" architecture is confirmed across phases: a smaller nodule shows more intense AP enhancement than the larger parent lesion, and this distinction often persists in later phases.'],

        # --- 6. 瘤周廓清 (Peripheral Washout) ---
        # 注释: “廓清”本身就需要AP和后续期相的对比。瘤周廓清是特指这种对比性减低发生在肿瘤的边缘。
        ['After initial AP enhancement, the lesion either does not show washout or shows a non-peripheral (diffuse) washout pattern in the PVP/DP.',
        'After initial AP enhancement, the lesion shows a distinct "peripheral washout" pattern, with only its rim becoming hypoenhancing in the PVP/DP.'],

        # --- 7. 延迟性中心强化 (Delayed Central Enhancement) ---
        # 注释: “延迟性”和“渐进性”强化，其定义就是基于对AP/PVP与DP中心区域信号强度变化的比较。
        ['Comparing phases, the central part of the lesion does not show progressive enhancement (e.g., it washes out or remains persistently non-enhancing).',
        'Comparing phases, the central part of the lesion shows progressive, sustained enhancement, becoming brighter in the delayed phase than it was in the AP/PVP.']
    ]

    # 2. 【【新的Prompt生成逻辑】】动态构建高级Prompt
    prompt_sections = []
    for i, name in enumerate(feature_names):
        option_0_desc = questions_list_professional[i][0]
        option_1_desc = questions_list_professional[i][1]
        prompt_sections.append(f"""
            --- Feature {i+1}: {name} ---
            Option 0 (Feature Absent): "{option_0_desc}"
            Option 1 (Feature Present): "{option_1_desc}"
            """)
    
    # 构建完整的、新的Prompt模板
    questions_prompt_template = f"""
You are a meticulous radiologist. Your task is to analyze a set of multi-phase CT images from a specific phase.
For each of the 7 features below, you are presented with two descriptive statements: Option 0 (first description) and Option 1 (second description).

Your tasks are:
1. Carefully analyze the provided images, comparing across different phases as needed.
2. For each feature, determine which option (0 or 1) most accurately describes the lesion.
3. Format your final output as a SINGLE, VALID JSON object. Do not add any explanatory text or markdown formatting outside of the JSON structure.

{''.join(prompt_sections)}

Your JSON output must follow this exact structure, containing all 7 features:
{{
  "Phase": "PHASE_ID",
  "findings": [
    {{"feature": "Enhancing Capsule", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Peritumoral Perfusion Alteration", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Corona Enhancement", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Fade Enhancement Pattern", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Nodule-in-Nodule Architecture", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Peripheral Washout", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}},
    {{"feature": "Delayed Central Enhancement", "value": <0_or_1>, "evidence": "Provide a brief clinical justification for your choice here."}}
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
        prompt = questions_prompt_template.replace("phase_ID", phase)
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

def run_phases_2_and_3(reports_from_phase1):
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
    Your sole task is to analyze these reports, focusing on the time dimension to summarize the evolution of each feature. Your output will be a concise, feature-centric evolution report **for the final Chief Radiologist**.
    It is important to highlight any inconsistencies in the findings between the different phase reports if they exist.
    """
    agent_4a = Agent(instruction=longitudinal_analyst_prompt, role="Longitudinal Analyst")
    longitudinal_report = agent_4a.chat(prompt_text=committee_input)
    cprint("--- Longitudinal Analyst Output ---", 'green')
    print(longitudinal_report)

    # Agent 4b: 横向分析师
    cross_sectional_analyst_prompt = """
    You are a senior radiologist acting as a **pattern recognition specialist** within an AI diagnostic committee.
    Your input consists of three structured reports generated by phase-specific AI analysts (AP, DP, PVP).
    Your sole task is to analyze these reports, focusing on each phase independently to provide a 'diagnostic snapshot'.
    Evaluate the combination of features present to determine the clinical picture for each phase. Your output will be a concise, timepoint-centric snapshot report **for the final Chief Radiologist**.
    """    
    agent_4b = Agent(instruction=cross_sectional_analyst_prompt, role="Cross-sectional Analyst")
    cross_sectional_report = agent_4b.chat(prompt_text=committee_input)
    cprint("--- Cross-sectional Analyst Output ---", 'green')
    print(cross_sectional_report)

    cprint("\n--- [Phase 3: Final Synthesized Diagnosis] ---", 'yellow', attrs=['bold'])
    
    synthesis_input = f"""
    [Feature Evolution Report from Longitudinal Analyst]:
    {longitudinal_report}
    [Diagnostic Snapshot Report from Cross-sectional Analyst]:
    {cross_sectional_report}
    """

    # Agent 5: 首席整合官
    chief_synthesizer_prompt = """
    You are the **Chief Radiologist** presiding over an AI diagnostic committee. Your task is to provide the final, global conclusion.
    Your input consists of two expert summaries, which were generated by specialist AI analysts:
    1.  A **Feature Evolution Report** from the Longitudinal Analyst.
    2.  A **Diagnostic Snapshot Report** from the Cross-sectional Analyst.
    Your job is to **synthesize these two distinct perspectives** (the 'what changed over time' and the 'what is the pattern now') to form a single, coherent, final report.

    Your final output MUST be a single block of text that strictly follows the three-part structure outlined below. It is CRITICAL that you include the exact headings for each section, including the numbering.

    1.  **Core Conclusion:** 
        A concise paragraph summarizing the overall clinical findings and conclusion.

    2.  **Main Evidence:**
        Bulleted points detailing the key evidence from both the longitudinal and cross-sectional reports that support your core conclusion.

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
        - "answer": The final binary conclusion (0 for absent, 1 for present).
        - "justification": A brief, concise summary of the reasoning.

    Example of the required structure for the JSON part ONLY:
    {
        "pattern_1": { "answer": 0, "justification": "Consistently absent" },
        "pattern_2": { "answer": 1, "justification": "Appeared in PVP" }
    }
    """

    agent_5 = Agent(instruction=chief_synthesizer_prompt, role="Chief Synthesizer")
    final_hybrid_output = agent_5.chat(prompt_text=synthesis_input)
    
    # 【【新】】增加解析逻辑，分离文本报告和JSON对象
    prose_report = ""
    structured_summary = {}
    
    try:
        # 我们使用 "Structured Summary:" 作为分割点
        if "Structured Summary:" in final_hybrid_output:
            parts = final_hybrid_output.split("Structured Summary:", 1)
            prose_report = parts[0].strip()
            json_text = parts[1].strip()
            
            # 清理并解析JSON
            clean_json_text = json_text.replace("```json", "").replace("```", "").strip()
            structured_summary = json.loads(clean_json_text)
        else:
            # 如果模型没有按预期输出，则将全部内容视为文本报告
            prose_report = final_hybrid_output
            structured_summary = {"error": "Structured Summary section not found in the output."}

    except Exception as e:
        cprint(f"Error parsing hybrid output from Chief Synthesizer: {e}", "red")
        prose_report = final_hybrid_output # 保留原始输出以供调试
        structured_summary = {"error": f"Failed to parse JSON part. Details: {e}"}

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
        phase1_reports = run_phase1_analysis(image_paths_ap, image_paths_dp, image_paths_pvp)
        print(f"phase1 reports: {phase1_reports}")
        
        # 执行第二和第三阶段
        prose_report, structured_summary = run_phases_2_and_3(phase1_reports)

        
        # 收集结果
        all_patient_results[patient_id] = structured_summary

        cprint(f"\n--- Final Hybrid Report for Patient {patient_id} ---", 'cyan', attrs=['bold'])
        print("--- Prose Report ---")
        print(prose_report)
        print("\n--- Structured Summary (JSON) ---")
        print(json.dumps(structured_summary, indent=4, ensure_ascii=False))

    # 将所有病人的结果保存到一个JSON文件中
    output_filename = "multi_agent_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_patient_results, f, ensure_ascii=False, indent=4)
    
    cprint(f"\n\n🎉 All processing complete. All reports have been saved to '{output_filename}'.", 'cyan', attrs=['bold'])


if __name__ == '__main__':
    main()
