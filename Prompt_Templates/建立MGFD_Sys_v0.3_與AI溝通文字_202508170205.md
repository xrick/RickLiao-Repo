# MGFD 系統架構分析報告

**報告日期**: 2025-08-17 01:49
**系統版本**: MGFD SalesRAG Integration System v0.3
**分析範圍**: 完整系統架構、模組依賴、配置文件使用分析


## 1. 系統模組組成

### 1.1 核心模組架構


#### **主控制器層**

```
MGFDSystem (mgfd_system.py)
├── 系統初始化和協調
├── 工作流程管理
└── 統一接口提供
```


#### **輸入處理層**

```
UserInputHandler (user_input_handler.py)
├── 用戶輸入解析
├── 槽位提取協調
└── 狀態更新管理
```


#### **狀態管理層**

```
RedisStateManager (redis_state_manager.py)
├── 會話狀態持久化
├── 對話歷史管理
├── 槽位信息存儲
└── 數據類型轉換
```


#### **回應生成層**

```
ResponseGenerator (response_generator.py)
├── 回應格式化
├── 前端渲染信息
├── 多類型回應處理
└── JSON序列化
```


#### **智能提取層**

```
EnhancedSlotExtractor (enhanced_slot_extractor.py)
├── 傳統關鍵詞匹配
├── LLM智能分類
├── 特殊案例處理
└── 置信度評估
```


#### **知識庫層**

```
NotebookKnowledgeBase (knowledge_base.py)
├── 產品數據管理
├── 搜索和過濾
├── 語義搜索
└── 推薦引擎
```


#### **狀態機層**

```
MGFDStateMachine (state_machine.py)
├── 狀態轉換管理
├── 流程控制
└── 事件處理
```

## 1.2 模組功能詳細分析

### 所有模組運作總述：

我們的目標是建立能夠服務客戶，回答客戶問題，幫客戶想到未能想到的需求，同時又能夠做知識庫，讓客戶能查詢所有產品的規格。

我會先用一個案例來描述這個系統的完整且符合預期的行為。

case 1:
necessary document: default_slot_doc.json

user-input1 -> [輸入處理層] -> reference default slot 