"""Prompt templates, Chinese — the set used for the experiments reported in the paper.

This module is the **experiment of record**. The policy model evaluated in the
paper was instruction-tuned on Chinese reasoning traces, so its behaviour is
conditioned on these exact strings; the English set in :mod:`autods.prompts_en`
is a translation provided for readability and is *not* what produced the
reported numbers.

The wording is reproduced verbatim from the research script, with two
mechanical changes and nothing else:

1. Values interpolated at call time became named ``str.format`` placeholders
   (``{data_root}``, ``{submission}``, ``{code}`` and so on). The rendered text
   is byte-identical to what the original f-strings produced.
2. In :data:`CODE_CORRECTION_SYSTEM_PROMPT`, ``{competition_name}`` is now a
   real placeholder. In the original it sat inside a plain (non-f) string
   literal, so the auxiliary repair model was shown the uninterpolated text
   ``'../input/{competition_name}/'``. See the fix table in the README.

Do not "tidy up" the phrasing here. Every edit invalidates the correspondence
between this repository and the reported results.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

#: Used for the first generation of every attempt. Fixes the think/answer output
#: contract that the code extractor depends on.
GENERATION_SYSTEM_PROMPT = (
    "你是一個嚴謹的 AI 助手，必須嚴格按照以下格式回答：\n"
    "1. 思考過程用 <think> 和 </think> 包裹，使用口語化中文逐步分析\n"
    "2. 最終答案用 <answer> 和 </answer> 包裹\n\n"
    "示範格式：\n"
    "<think>\n"
    "好的，我需要解決這個問題。首先確認...\n"
    "等等，如果遇到...該怎麼辦？接著應該...\n"
    "比方說...，最後整合所有條件判斷。\n"
    "</think>\n\n"
    "<answer>\n最終答案放在這裡\n</answer>\n\n"
    "請注意：\n"
    "- 以『好的』、『嗯』等語氣詞開頭\n"
    "- 使用『首先』、『接著』、『然後』銜接步驟\n"
    "- 用『等等』引導補充說明或疑問\n"
    "現在開始回答用戶問題："
)

#: Used when the previous attempt executed successfully and the goal is to
#: improve the score rather than to fix a failure.
OPTIMIZATION_SYSTEM_PROMPT = (
    "你是一位專業的機器學習工程師，精通 Kaggle 競賽解決方案。請基於以下信息優化現有方案：\n"
    "1. 分析之前方案的性能指標和錯誤\n"
    "2. 利用原始思考過程中的洞見\n"
    "3. 思考如何改進模型架構、特徵工程或參數調整\n"
    "4. 生成一個完整且能直接執行的 Python 代碼\n\n"
    "回答格式：\n"
    "<think>\n詳細分析現有方案的問題和可能的改進點\n</think>\n\n"
    "<answer>\n完整的優化代碼\n</answer>"
)

#: Terse system prompt for the auxiliary repair model: code only, no reasoning.
CODE_CORRECTION_SYSTEM_PROMPT = (
    "你是 Python 代碼修正專家。要求：\n"
    "1. 分析錯誤並修正代碼\n"
    "2. 只返回完整的修正後代碼，無需解釋\n"
    "3. 代碼必須完整可執行\n"
    "4. 確保生成 {submission}\n"
    "5. 不使用 try...except 包裹提交文件生成\n"
    "6. **重要**: 嚴格從 '{data_root}' 讀取數據，禁止自創/模擬數據\n\n"
    "格式：直接返回修正後的 Python 代碼。"
)

#: System prompt for the auxiliary model when summarising a failure history.
ERROR_ANALYSIS_SYSTEM_PROMPT = "你是一位效率專家，擅長簡潔精確地分析代碼錯誤並提供要點建議。"


# --------------------------------------------------------------------------- #
# User prompt templates
# --------------------------------------------------------------------------- #

TIMEOUT_OPTIMIZATION_TEMPLATE = """以下代碼執行時發生超時問題，需要優化以提高執行效率：

錯誤信息：
{error_message}

原始代碼：
```python
{code}
```

請優化代碼以避免超時，可以考慮以下策略：
1. 簡化深度學習網路架構（減少層數、神經元數量）
2. 減少訓練epoch數或batch size
3. 使用更簡單的機器學習算法
4. 優化數據預處理流程
5. 減少特徵數量或樣本數量
6. 使用更高效的算法實現

要求：
- 數據路徑：{data_root}
- 必須生成{submission}文件
- 代碼必須能在合理時間內完成執行
- 不使用try...except包裹提交文件生成邏輯

請直接返回完整的優化後Python代碼。"""


GENERIC_CORRECTION_TEMPLATE = """錯誤：{error_tail}

修正以下代碼並返回完整可執行版本：
```python
{code}
```

要求：數據在{data_root}，生成{submission}"""


ERROR_ANALYSIS_TEMPLATE = """分析以下Python代碼執行錯誤，用簡潔的要點列出關鍵問題和解決建議：

{errors_text}

請簡要回答：
1. 共同錯誤模式（最多3點，每點30字以內）
2. 主要問題類型（最多2種，每種15字以內）
3. 關鍵解決建議（最多4點，每點25字以內）

格式要求：
- 不要有多餘的解釋
- 以簡短的要點列表呈現
- 總回應不超過300字
- 純文字輸出，無需標記"""


IMPROVED_TASK_TEMPLATE = """# 數據科學任務：{competition_name}

## 任務描述與要求
{original_prompt}

## 實現要點
請提供一個完整的解決方案，達成上述數據科學的目標。你的代碼必須：

1. 正確理解並完成任務核心目標
2. 從 '{data_root}' 讀取數據
3. 生成符合競賽要求的{submission}文件
4. 遵循數據科學最佳實踐，包括數據預處理、特徵工程和模型選擇

## 避免常見錯誤
之前的嘗試中發現以下問題，請在設計解決方案時避免：

{error_analysis}

## 代碼要求
- 提供完整可執行的Python代碼，包括所有必要的導入語句
- 添加清晰的註釋解釋關鍵步驟及決策理由
- 優先使用穩定可靠的方法而非高風險實驗性技術
- 確保代碼高效執行，不會超時或耗盡記憶體
- 【重要】絕對不能使用 try...except 來包裹提交文件生成邏輯，必須確保能正確產生 {submission}

請注意：首要目標是完成數據科學任務並產生有效預測，同時代碼必須穩定可靠地執行。
"""


REGENERATION_TEMPLATE = """請為以下數據科學任務重新生成一個完整的解決方案。

## 原始任務說明
{original_question}

## 重新生成原因
{reason_text}，需要重新設計解決方案。

## 數據位置
{data_root}

## 解決方案要求
{focus_text}，確保能正確運行並生成{submission}文件。

### 具體要求：
1. **代碼完整性**：提供完整可執行的Python代碼，包含所有必要的import語句
2. **錯誤處理**：處理所有可能的邊界情況和異常
3. **輸出格式**：生成正確格式的{submission}文件
4. **代碼註釋**：包含必要的註釋解釋關鍵步驟和決策理由
5. **技術選擇**：優先使用最基礎穩定的庫和方法，避免過於複雜的實現
6. **性能考慮**：確保代碼能在合理時間內完成執行
7. **【重要】**：絕對不能使用 try...except 來包裹提交文件生成邏輯，必須確保能正確產生 {submission}

### 實現策略：
- 採用經典且被驗證的機器學習方法
- 使用簡潔明瞭的數據預處理流程
- 確保每個步驟都有清晰的邏輯和目標
- 避免過度工程化，專注於解決核心問題

請依據原始任務說明深入思考解決方法，然後提供完整可執行的Python代碼。
"""


OPTIMIZATION_TEMPLATE = """請優化以下數據科學任務的解决方案。

### 任务描述:
{original_prompt}

### 當前狀況:
{metrics_info}

### 原始思考过程:
{thinking}

請分析當前解決方案並提出改進方案。重點關注:
1. 特徵工程是否充分
2. 模型選擇是否恰當
3. 超參數是否需要調整
4. 是否有潛在的數據洩漏問題
5. 是否需要集成多個模型
6. 交叉驗證策略是否合理

請生成一個完整且更高效的Python代碼，確保與現有代碼有顯著差異，並且能夠提高模型性能。
代碼必須完整可執行，注意數據路徑為 "{data_root}"，並輸出Kaggle要求的{submission}文件。

特別注意：絕對不能使用 try...except 來包裹提交文件生成邏輯，必須確保能正確產生 {submission}。
"""


METRICS_KNOWN_TEMPLATE = """當前模型在Kaggle上的表現:
- 公開測試集分數: {public_score}
- 私有測試集分數: {private_score}

請分析這些分數並思考如何改進以獲得更好的排名。
"""


METRICS_UNKNOWN_TEMPLATE = """由於無法獲取先前模型的具體評分指標，請專注於以下優化策略：
- 改進特徵工程技術
- 嘗試更先進的模型架構
- 優化超參數設置
- 考慮資料預處理的改進空間

"""


#: Why a solution is being regenerated, keyed by the machine-readable reason
#: recorded in the attempt record. Each value is ``(reason_text, focus_text)``.
REGENERATION_REASON_TEXT = {
    "correction_failed": (
        "之前的代碼在執行過程中遇到了反復出現的問題",
        "請生成一個更簡單、更健壯的解決方案",
    ),
    "extract_failed": (
        "之前的代碼提取失敗，需要提高創造性",
        "請生成一個更創新、格式清晰的解決方案",
    ),
}

DEFAULT_REGENERATION_REASON_TEXT = (
    "之前的代碼在執行過程中遇到了問題",
    "請生成一個更穩定可靠的解決方案",
)
